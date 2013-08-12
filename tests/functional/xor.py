#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

xor with rbm.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import logging
import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s" % (this_dir))
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import units
import formats
import error
import numpy
import config
import rnd
import opencl
import plotters
import pickle
import time
import rbm
import mnist_ae


def normalize(a):
    a -= a.min()
    m = a.max()
    if m:
        a /= m
        a *= 2.0
        a -= 1.0


class Loader(units.Unit):
    """Loads MNIST data and provides mini-batch output interface.

    Attributes:
        rnd: rnd.Rand().
        use_hog: use hog or not.

        minibatch_data: MNIST images scaled to [-1, 1].
        minibatch_indexes: global indexes of images in minibatch.
        minibatch_labels: labels for indexes in minibatch.

        minibatch_class: class of the minibatch: 0-test, 1-validation, 2-train.
        minibatch_last: if current minibatch is last in it's class.

        minibatch_offs: offset of the current minibatch in all samples,
                        where first come test samples, then validation, with
                        train ones at the end.
        minibatch_size: size of the current minibatch.
        total_samples: total number of samples in the dataset.
        class_samples: number of samples per class.
        minibatch_maxsize: maximum size of minibatch in samples.
        nextclass_offs: offset in samples where the next class begins.

        original_data: original MNIST images scaled to [-1, 1] as single batch.
        original_labels: original MNIST labels as single batch.
    """
    def __init__(self, classes=[0, 0, 8], minibatch_max_size=6,
                 rnd=rnd.default, use_hog=False):
        """Constructor.

        Parameters:
            classes: [test, validation, train],
                ints - in samples,
                floats - relative from (0 to 1).
            minibatch_size: minibatch max size.
        """
        super(Loader, self).__init__()
        self.rnd = [rnd]
        self.use_hog = use_hog

        self.minibatch_data = formats.Batch()
        self.minibatch_indexes = formats.Labels(8)
        self.minibatch_labels = formats.Labels(2)

        self.minibatch_class = [0]
        self.minibatch_last = [0]

        self.total_samples = [8]
        self.class_samples = classes.copy()
        if type(self.class_samples[2]) == float:
            smm = 0
            for i in range(0, len(self.class_samples) - 1):
                self.class_samples[i] = int(
                numpy.round(self.total_samples[0] * self.class_samples[i]))
                smm += self.class_samples[i]
            self.class_samples[-1] = self.total_samples[0] - smm
        self.minibatch_offs = [self.total_samples[0]]
        self.minibatch_size = [0]
        self.minibatch_maxsize = [minibatch_max_size]
        self.nextclass_offs = [0, 0, 0]
        offs = 0
        for i in range(0, len(self.class_samples)):
            offs += self.class_samples[i]
            self.nextclass_offs[i] = offs
        if self.nextclass_offs[-1] != self.total_samples[0]:
            raise error.ErrBadFormat("Sum of class samples (%d) differs from "
                "total number of samples (%d)" % (self.nextclass_offs[-1],
                                                  self.total_samples[0]))

        self.original_data = None
        self.original_labels = None

        self.shuffled_indexes = None

    def initialize(self):
        """Here we will load MNIST data.
        """
        self.original_labels = numpy.array(
            [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1], dtype=numpy.int8)
        self.original_data = numpy.array([
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]],
                dtype=config.dtypes[config.dtype])
        self.shuffled_indexes = numpy.arange(len(self.original_labels),
                                             dtype=numpy.int8)

        self.original_data[:] = numpy.where(self.original_data == 0, -1.0,
                                            1.0)[:]

        self.minibatch_indexes.n_classes = len(self.original_labels)
        self.class_samples[2] = len(self.original_labels)
        self.total_samples[0] = len(self.original_labels)

        offs = 0
        for i in range(0, len(self.class_samples)):
            offs += self.class_samples[i]
            self.nextclass_offs[i] = offs
        if self.nextclass_offs[-1] != self.total_samples[0]:
            raise error.ErrBadFormat("Sum of class samples (%d) differs from "
                "total number of samples (%d)" % (self.nextclass_offs[-1],
                                                  self.total_samples[0]))

        sh = [self.minibatch_maxsize[0]]
        for i in self.original_data.shape[1:]:
            sh.append(i)
        self.minibatch_data.batch = numpy.zeros(
            sh, dtype=config.dtypes[config.dtype])
        self.minibatch_labels.v = numpy.zeros(
            [self.minibatch_maxsize[0]], dtype=numpy.int8)
        self.minibatch_indexes.batch = numpy.zeros(
            [self.minibatch_maxsize[0]], dtype=numpy.int8)

        if self.class_samples[0]:
            self.shuffle_validation_train()
        else:
            self.shuffle_train()

    def shuffle_validation_train(self):
        """Shuffles validation and train dataset
            so the layout will be:
                0:10000: test,
                10000:20000: randomized validation,
                20000:70000: randomized train.
        """
        self.rnd[0].shuffle(self.shuffled_indexes[self.nextclass_offs[0]:\
                                                  self.nextclass_offs[2]])

    def shuffle_train(self):
        """Shuffles train dataset
            so the layout will be:
                0:10000: test,
                10000:20000: validation,
                20000:70000: randomized train.
        """
        self.rnd[0].shuffle(self.shuffled_indexes[self.nextclass_offs[1]:\
                                                  self.nextclass_offs[2]])

    def shuffle(self):
        """Shuffle the dataset after one epoch.
        """
        self.shuffle_train()

    def run(self):
        """Prepare the minibatch.
        """
        t1 = time.time()

        self.minibatch_offs[0] += self.minibatch_size[0]
        # Reshuffle when end of data reached.
        if self.minibatch_offs[0] >= self.total_samples[0]:
            self.shuffle()
            self.minibatch_offs[0] = 0

        # Compute minibatch size and it's class.
        for i in range(0, len(self.nextclass_offs)):
            if self.minibatch_offs[0] < self.nextclass_offs[i]:
                self.minibatch_class[0] = i
                minibatch_size = min(self.minibatch_maxsize[0],
                    self.nextclass_offs[i] - self.minibatch_offs[0])
                if self.minibatch_offs[0] + minibatch_size >= \
                   self.nextclass_offs[self.minibatch_class[0]]:
                    self.minibatch_last[0] = 1
                else:
                    self.minibatch_last[0] = 0
                break
        else:
            raise error.ErrNotExists("Could not determine minibatch class.")
        self.minibatch_size[0] = minibatch_size

        # Fill minibatch data labels and indexes according to current shuffle.
        idxs = self.minibatch_indexes.batch
        idxs[0:minibatch_size] = self.shuffled_indexes[self.minibatch_offs[0]:\
            self.minibatch_offs[0] + minibatch_size]

        self.minibatch_labels.v[0:minibatch_size] = \
            self.original_labels[idxs[0:minibatch_size]]

        self.minibatch_data.batch[0:minibatch_size] = \
            self.original_data[idxs[0:minibatch_size]]

        # Fill excessive indexes.
        if minibatch_size < self.minibatch_maxsize[0]:
            self.minibatch_data.batch[minibatch_size:] = 0.0
            self.minibatch_labels.v[minibatch_size:] = -1
            self.minibatch_indexes.batch[minibatch_size:] = -1

        # Set update flag for GPU operation.
        self.minibatch_data.update()
        self.minibatch_labels.update()
        self.minibatch_indexes.update()

        self.log().debug("%s in %.2f sec" % (self.__class__.__name__,
                                             time.time() - t1))


import all2all
import evaluator
import gd


class Decision(units.Unit):
    """Decides on the learning behavior.

    Attributes:
        complete: completed.
        minibatch_class: current minibatch class.
        minibatch_last: if current minibatch is last in it's class.
        gd_skip: skip gradient descent or not.
        epoch_number: epoch number.
        epoch_min_mse: minimum sse by class per epoch.
        minibatch_n_err: number of errors for minibatch.
        minibatch_metrics: [0] - sse, [1] - max of sum of sample graidents.
        class_samples: number of samples per class.
        epoch_ended: if an epoch has ended.
        fail_iterations: number of consequent iterations with non-decreased
            validation error.
        epoch_metrics: metrics for each set epoch.
    """
    def __init__(self, fail_iterations=1000):
        super(Decision, self).__init__()
        self.complete = [0]
        self.minibatch_class = None  # [0]
        self.minibatch_last = None  # [0]
        self.gd_skip = [0]
        self.epoch_number = [0]
        self.epoch_min_mse = [1.0e30, 1.0e30, 1.0e30]
        self.n_err = [1.0e30, 1.0e30, 1.0e30]
        self.minibatch_n_err = None  # formats.Vector()
        self.minibatch_metrics = None  # formats.Vector()
        self.fail_iterations = [fail_iterations]
        self.epoch_ended = [0]
        self.n_err_pt = [100.0, 100.0, 100.0]
        self.class_samples = None  # [0, 0, 0]
        self.min_validation_mse = 1.0e30
        self.min_validation_mse_epoch_number = -1
        self.prev_train_err = 1.0e30
        self.workflow = None
        self.fnme = None
        self.t1 = None
        self.epoch_metrics = [None, None, None]
        self.just_snapshotted = [0]
        self.snapshot_date = [0]
        self.threshold_ok = 0.0
        self.weights_to_sync = []
        self.sample_output = None
        self.sample_input = None
        self.all_mse = [formats.Vector(), formats.Vector(), formats.Vector()]
        self.mse = [formats.Vector(), formats.Vector(), formats.Vector()]
        self.minibatch_mse = None
        self.minibatch_offs = None
        self.minibatch_size = None

    def initialize(self):
        if (self.minibatch_metrics != None and
            self.minibatch_metrics.v != None):
            for i in range(0, len(self.epoch_metrics)):
                self.epoch_metrics[i] = (
                    numpy.zeros_like(self.minibatch_metrics.v))
        self.sample_output = numpy.zeros_like(
            self.workflow.forward[-1].output.batch[0])
        self.sample_input = numpy.zeros_like(
            self.workflow.forward[0].input.batch[0])
        for i in range(0, len(self.mse)):
            if self.class_samples[i] <= 0:
                continue
            if (self.mse[i].v == None or
                self.mse[i].v.size != self.class_samples[i]):
                self.mse[i].v = numpy.zeros(self.class_samples[i],
                                         dtype=config.dtypes[config.dtype])
                self.all_mse[i].v = numpy.zeros(self.class_samples[i],
                                         dtype=config.dtypes[config.dtype])

    def run(self):
        if self.t1 == None:
            self.t1 = time.time()
        self.complete[0] = 0
        self.epoch_ended[0] = 0

        minibatch_class = self.minibatch_class[0]

        if self.minibatch_last[0]:
            self.minibatch_metrics.sync()
            if self.epoch_number[0] > 5:
                self.epoch_min_mse[minibatch_class] = (
                    min(self.minibatch_metrics.v[0] /
                    self.class_samples[minibatch_class],
                    self.epoch_min_mse[minibatch_class]))

            self.minibatch_n_err.sync()
            self.n_err[minibatch_class] = self.minibatch_n_err.v[0]

            # Compute error in percents
            if self.class_samples[minibatch_class]:
                self.n_err_pt[minibatch_class] = (100.0 *
                    self.n_err[minibatch_class] /
                    self.class_samples[minibatch_class])

        self.minibatch_mse.sync(read_only=True)
        offs = self.minibatch_offs[0]
        for i in range(0, minibatch_class):
            offs -= self.class_samples[i]
        size = self.minibatch_size[0]
        self.mse[minibatch_class].v[offs:offs + size] = \
            self.minibatch_mse.v[:size]

        # Check skip gradient descent or not
        if self.minibatch_class[0] < 2:
            self.gd_skip[0] = 1
        else:
            self.gd_skip[0] = 0

        if self.minibatch_last[0]:
            self.epoch_metrics[minibatch_class][:] = (
                self.minibatch_metrics.v[:])
            self.epoch_metrics[minibatch_class][0] = (
                self.epoch_metrics[minibatch_class][0] /
                self.class_samples[minibatch_class])

            # Test and Validation sets processed
            if self.minibatch_class[0] >= 1:
                if self.just_snapshotted[0]:
                    self.just_snapshotted[0] = 0
                if (self.epoch_min_mse[minibatch_class] <
                    self.min_validation_mse):
                    self.min_validation_mse = self.epoch_min_mse[
                        minibatch_class]
                    self.min_validation_mse_epoch_number = self.epoch_number[0]
                    if self.epoch_metrics[minibatch_class][0] < 0.5:
                        if self.fnme != None:
                            try:
                                os.unlink(self.fnme)
                            except FileNotFoundError:
                                pass
                        self.fnme = "%s/xor.%.6f.pickle" % \
                            (config.snapshot_dir,
                             self.epoch_metrics[minibatch_class][0])
                        self.log().info("Snapshotting to %s" % (self.fnme))
                        fout = open(self.fnme, "wb")
                        pickle.dump(self.workflow, fout)
                        fout.close()
                        self.just_snapshotted[0] = 1
                        self.snapshot_date[0] = time.time()
                # Stop condition
                if self.epoch_number[0] - \
                   self.min_validation_mse_epoch_number > \
                   self.fail_iterations[0]:
                    self.complete[0] = 1

            # Print some statistics
            t2 = time.time()
            self.log().info(
                "Epoch %d Class %d AvgMSE %.6f Greater%.3f %d (%.2f%%) "
                "MaxMSE %.6f MinMSE %.2e in %.2f sec" % \
                (self.epoch_number[0], self.minibatch_class[0],
                 self.epoch_metrics[self.minibatch_class[0]][0],
                 self.threshold_ok,
                 self.n_err[self.minibatch_class[0]],
                 self.n_err_pt[self.minibatch_class[0]],
                 self.epoch_metrics[self.minibatch_class[0]][1],
                 self.epoch_metrics[self.minibatch_class[0]][2],
                 t2 - self.t1))
            self.t1 = t2

            # Training set processed
            if self.minibatch_class[0] == 2:
                # """
                this_train_err = self.epoch_metrics[2][0]
                if self.prev_train_err:
                    k = this_train_err / self.prev_train_err
                else:
                    k = 1.0
                if k < 1.04:
                    ak = 1.05
                else:
                    ak = 0.7
                self.prev_train_err = this_train_err
                for gd in self.workflow.gd:
                    gd.global_alpha = max(min(ak * gd.global_alpha, 0.99999),
                                          0.00001)
                self.log().info("new global_alpha: %.4f" % \
                      (self.workflow.gd[0].global_alpha))
                # """
                self.epoch_ended[0] = 1
                self.epoch_number[0] += 1
                # Reset n_err
                for i in range(0, len(self.n_err)):
                    self.n_err[i] = 0
                # Sync weights
                for weights in self.weights_to_sync:
                    weights.sync(read_only=True)
                self.workflow.forward[0].input.sync(read_only=True)
                self.workflow.forward[-1].output.sync(read_only=True)
                self.sample_output[:] = \
                    self.workflow.forward[-1].output.batch[0][:]
                self.sample_input[:] = \
                    self.workflow.forward[0].input.batch[0][:]

            # Reset statistics per class
            self.minibatch_n_err.v[:] = 0
            self.minibatch_n_err.update()
            if (self.minibatch_metrics != None and
                self.minibatch_metrics.v != None):
                self.minibatch_metrics.v[:] = 0
                self.minibatch_metrics.v[2] = 1.0e30
                self.minibatch_metrics.update()
            if (self.all_mse[minibatch_class] != None and
                self.all_mse[minibatch_class].v != None):
                self.all_mse[minibatch_class].v[:] = \
                    self.mse[minibatch_class].v[:]
            self.mse[minibatch_class].v[:] = 0


class Workflow(units.OpenCLUnit):
    """Sample workflow for MNIST dataset.

    Attributes:
        start_point: start point.
        rpt: repeater.
        loader: loader.
        forward: list of all-to-all forward units.
        ev: evaluator softmax.
        stat: stat collector.
        decision: Decision.
        gd: list of gradient descent units.
    """
    def __init__(self, layers=None, device=None):
        super(Workflow, self).__init__(device=device)
        self.start_point = units.Unit()

        self.rpt = units.Repeater()
        self.rpt.link_from(self.start_point)

        self.loader = Loader()
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward = []
        for i in range(0, len(layers)):
            if not i:
                amp = None
            else:
                amp = 9.0 / 1.7159 / layers[i - 1]
            if not i:
                aa = rbm.RBMTanh([layers[i]], device=device,
                             weights_amplitude=amp)
            else:
                aa = all2all.All2AllTanh([layers[i]], device=device,
                             weights_amplitude=amp,
                             weights_transposed=True)
                aa.weights = self.forward[0].weights
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorMSE(device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.target = self.loader.minibatch_data
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = Decision()
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err_skipped
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.minibatch_mse = self.ev.mse
        self.decision.minibatch_offs = self.loader.minibatch_offs
        self.decision.minibatch_size = self.loader.minibatch_size
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add Image Saver unit
        self.image_saver = mnist_ae.ImageSaverAE(["/tmp/img/test",
                                                  "/tmp/img/validation",
                                                  "/tmp/img/train"])
        self.image_saver.link_from(self.decision)
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.forward[-1].output
        self.image_saver.indexes = self.loader.minibatch_indexes
        self.image_saver.labels = self.loader.minibatch_labels
        self.image_saver.minibatch_class = self.loader.minibatch_class
        self.image_saver.minibatch_size = self.loader.minibatch_size
        self.image_saver.this_save_date = self.decision.snapshot_date
        self.image_saver.gate_skip = [0]  # self.decision.just_snapshotted
        self.image_saver.gate_skip_not = [1]

        # Add gradient descent units
        self.gd = list(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GD(device=device, weights_transposed=True)
        # self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            if False:
                self.gd[i] = gd.GD(device=device)
            elif i:
                self.gd[i] = gd.GDTanh(device=device)
            else:
                self.gd[i] = gd.GDTanh(device=device)
                # self.gd[i] = rbm.GDTanh(device=device,
                #                        rnd_window_size=1.0)
                # self.gd[i].y_rand = self.forward[i].output_rand
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
        self.rpt.link_from(self.gd[0])

        self.end_point = units.EndPoint()
        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(2, 3):
            self.plt.append(plotters.SimplePlotter(figure_label="mse",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if i == 2 else
                                   self.plt[-2])
            self.plt[-1].gate_skip = self.decision.epoch_ended
            self.plt[-1].gate_skip_not = [1]
        # self.plt[-1].clear_plot = True
        # Weights plotter
        self.decision.weights_to_sync.clear()
        self.decision.weights_to_sync.append(self.gd[0].weights)
        self.plt_mx = []
        self.plt_mx.append(
            plotters.Weights2D(figure_label="First Layer Weights", limit=16))
        self.plt_mx[-1].input = self.decision.weights_to_sync[-1]
        self.plt_mx[-1].input_field = "v"
        self.plt_mx[-1].get_shape_from = self.forward[0].input
        self.plt_mx[-1].link_from(self.decision)
        self.plt_mx[-1].gate_block = [0]  # self.decision.epoch_ended
        self.plt_mx[-1].gate_block_not = [1]
        # Weights plotter
        self.decision.weights_to_sync.append(self.gd[-1].weights)
        self.plt_mx.append(
            plotters.Weights2D(figure_label="Last Layer Weights", limit=16))
        self.plt_mx[-1].input = self.decision.weights_to_sync[-1]
        self.plt_mx[-1].input_field = "v"
        # self.plt_mx[-1].transposed = True
        self.plt_mx[-1].get_shape_from = self.forward[0].input
        self.plt_mx[-1].link_from(self.plt_mx[-2])
        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(2, 3):
            self.plt_max.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.plt[-1] if i == 2 else
                                       self.plt_max[-2])
            self.plt_max[-1].gate_skip = self.decision.epoch_ended
            self.plt_max[-1].gate_skip_not = [1]
        # self.plt_max[0].clear_plot = True
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(2, 3):
            self.plt_min.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.plt_max[-1] if i == 2 else
                                       self.plt_min[-2])
            self.plt_min[-1].gate_skip = self.decision.epoch_ended
            self.plt_min[-1].gate_skip_not = [1]
        # self.plt_min[0].clear_plot = True
        # Image plotter
        self.plt_img = plotters.Image(figure_label="output sample")
        self.plt_img.inputs.append(self.decision)
        self.plt_img.input_fields.append("sample_input")
        self.plt_img.inputs.append(self.decision)
        self.plt_img.input_fields.append("sample_output")
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_block = [0]  # self.decision.epoch_ended
        self.plt_img.gate_block_not = [1]
        # Histogram plotter
        self.plt_hist = [None, None, plotters.MSEHistogram(
                                        figure_label="Histogram Train")]
        self.plt_hist[2].link_from(self.plt_min[-1])
        self.plt_hist[2].mse = self.decision.all_mse[2]
        self.plt_hist[2].gate_skip = self.decision.epoch_ended
        self.plt_hist[2].gate_skip_not = [1]
        self.gd[-1].link_from(self.plt_hist[2])

    def initialize(self):
        retval = self.start_point.initialize_dependent()
        if retval:
            return retval

    def run(self, threshold_ok, threshold_skip, global_alpha, global_lambda):
        self.ev.threshold_ok = threshold_ok
        self.ev.threshold_skip = threshold_skip
        self.decision.threshold_ok = threshold_ok
        for gd in self.gd:
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
        retval = self.start_point.run_dependent()
        if retval:
            return retval
        self.end_point.wait()


# import scipy.misc


class Decision2(units.Unit):
    """Decides on the learning behavior.

    Attributes:
        complete: completed.
        minibatch_class: current minibatch class.
        minibatch_last: if current minibatch is last in it's class.
        gd_skip: skip gradient descent or not.
        epoch_number: epoch number.
        epoch_min_err: minimum number of errors by class per epoch.
        n_err: current number of errors per class.
        minibatch_n_err: number of errors for minibatch.
        minibatch_confusion_matrix: confusion matrix for minibatch.
        minibatch_max_err_y_sum: max last layer backpropagated errors sum.
        n_err_pt: n_err in percents.
        class_samples: number of samples per class.
        epoch_ended: if an epoch has ended.
        fail_iterations: number of consequent iterations with non-decreased
            validation error.
        confusion_mxs: confusion matrixes.
        max_err_y_sums: max last layer backpropagated errors sums for each set.
    """
    def __init__(self, fail_iterations=100):
        super(Decision2, self).__init__()
        self.complete = [0]
        self.minibatch_class = None  # [0]
        self.minibatch_last = None  # [0]
        self.gd_skip = [0]
        self.epoch_number = [0]
        self.epoch_min_err = [1.0e30, 1.0e30, 1.0e30]
        self.n_err = [0, 0, 0]
        self.minibatch_n_err = None  # formats.Vector()
        self.minibatch_confusion_matrix = None  # formats.Vector()
        self.minibatch_max_err_y_sum = None  # formats.Vector()
        self.fail_iterations = [fail_iterations]
        self.epoch_ended = [0]
        self.n_err_pt = [100.0, 100.0, 100.0]
        self.class_samples = None  # [0, 0, 0]
        self.min_validation_err = 1.0e30
        self.min_validation_err_epoch_number = -1
        self.prev_train_err = 1.0e30
        self.workflow = None
        self.fnme = None
        self.t1 = None
        self.confusion_matrixes = [None, None, None]
        self.max_err_y_sums = [None, None, None]

    def initialize(self):
        if (self.minibatch_confusion_matrix != None and
            self.minibatch_confusion_matrix.v != None):
            for i in range(0, len(self.confusion_matrixes)):
                self.confusion_matrixes[i] = (
                    numpy.zeros_like(self.minibatch_confusion_matrix.v))
        if (self.minibatch_max_err_y_sum != None and
            self.minibatch_max_err_y_sum.v != None):
            for i in range(0, len(self.max_err_y_sums)):
                self.max_err_y_sums[i] = (
                    numpy.zeros_like(self.minibatch_max_err_y_sum.v))

    def run(self):
        if self.t1 == None:
            self.t1 = time.time()
        self.complete[0] = 0
        self.epoch_ended[0] = 0

        minibatch_class = self.minibatch_class[0]

        if self.minibatch_last[0]:
            self.minibatch_n_err.sync()
            self.n_err[minibatch_class] = self.minibatch_n_err.v[0]
            self.epoch_min_err[minibatch_class] = \
                min(self.n_err[minibatch_class],
                    self.epoch_min_err[minibatch_class])
            # Compute error in percents
            if self.class_samples[minibatch_class]:
                self.n_err_pt[minibatch_class] = 100.0 * \
                    self.n_err[minibatch_class] / \
                    self.class_samples[minibatch_class]

        # Check skip gradient descent or not
        if self.minibatch_class[0] < 2:
            self.gd_skip[0] = 1
        else:
            self.gd_skip[0] = 0

        if self.minibatch_last[0]:
            if (self.minibatch_confusion_matrix != None and
                self.minibatch_confusion_matrix.v != None):
                self.minibatch_confusion_matrix.sync()
                self.confusion_matrixes[minibatch_class][:] = (
                    self.minibatch_confusion_matrix.v[:])
            if (self.minibatch_max_err_y_sum != None and
                self.minibatch_max_err_y_sum.v != None):
                self.minibatch_max_err_y_sum.sync()
                self.max_err_y_sums[minibatch_class][:] = (
                    self.minibatch_max_err_y_sum.v[:])

            # Test and Validation sets processed
            if self.minibatch_class[0] >= 1:
                if (self.epoch_min_err[minibatch_class] <
                    self.min_validation_err):
                    self.min_validation_err = self.epoch_min_err[
                                                    minibatch_class]
                    self.min_validation_err_epoch_number = self.epoch_number[0]
                    if self.n_err_pt[minibatch_class] < 2.5:
                        if self.fnme != None:
                            try:
                                os.unlink(self.fnme)
                            except FileNotFoundError:
                                pass
                        self.fnme = "%s/xor.%.2f.pickle" % \
                            (config.snapshot_dir, self.n_err_pt[1])
                        self.log().info("Snapshotting to %s" % (self.fnme))
                        fout = open(self.fnme, "wb")
                        pickle.dump(self.workflow, fout)
                        fout.close()
                # Stop condition
                if self.epoch_number[0] - \
                   self.min_validation_err_epoch_number > \
                   self.fail_iterations[0]:
                    self.complete[0] = 1

            # Print some statistics
            t2 = time.time()
            self.log().info("Epoch %d Class %d Errors %d in %.2f sec" % \
                  (self.epoch_number[0], self.minibatch_class[0],
                   self.n_err[self.minibatch_class[0]],
                   t2 - self.t1))
            self.t1 = t2

            # Training set processed
            if self.minibatch_class[0] == 2:
                this_train_err = self.n_err[2]
                if self.prev_train_err:
                    k = this_train_err / self.prev_train_err
                else:
                    k = 1.0
                if k < 1.04:
                    ak = 1.05
                else:
                    ak = 0.7
                self.prev_train_err = this_train_err
                for gd in self.workflow.gd:
                    if gd == None:
                        continue
                    gd.global_alpha = max(min(ak * gd.global_alpha, 0.9999),
                                          0.0001)
                self.log().info("new global_alpha: %.4f" % \
                    (self.workflow.gd[-1].global_alpha))

                self.epoch_ended[0] = 1
                self.epoch_number[0] += 1
                # Reset n_err
                for i in range(0, len(self.n_err)):
                    self.n_err[i] = 0

            # Reset statistics per class
            self.minibatch_n_err.v[:] = 0
            self.minibatch_n_err.update()
            if (self.minibatch_confusion_matrix != None and
                self.minibatch_confusion_matrix.v != None):
                self.minibatch_confusion_matrix.v[:] = 0
                self.minibatch_confusion_matrix.update()
            if (self.minibatch_max_err_y_sum != None and
                self.minibatch_max_err_y_sum.v != None):
                self.minibatch_max_err_y_sum.v[:] = 0
                self.minibatch_max_err_y_sum.update()


class Workflow2(units.OpenCLUnit):
    """Sample workflow for MNIST dataset.

    Attributes:
        start_point: start point.
        rpt: repeater.
        loader: loader.
        forward: list of all-to-all forward units.
        ev: evaluator softmax.
        stat: stat collector.
        decision: Decision.
        gd: list of gradient descent units.
    """
    def __init__(self, layers=None, device=None):
        super(Workflow2, self).__init__(device=device)
        self.start_point = units.Unit()

        self.rpt = units.Repeater()
        self.rpt.link_from(self.start_point)

        self.loader = Loader()
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward = []
        for i in range(0, len(layers)):
            if not i:
                amp = None
            else:
                amp = 9.0 / 1.7159 / layers[i - 1]
            if i < len(layers) - 1:
                if not i:
                    aa = rbm.RBMTanh([layers[i]], device=device,
                                 weights_amplitude=amp)
                else:
                    aa = all2all.All2AllTanh([layers[i]], device=device,
                                 weights_amplitude=amp)
            else:
                aa = all2all.All2AllSoftmax([layers[i]], device=device,
                                            weights_amplitude=amp)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_idx = self.forward[-1].max_idx
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = Decision2()
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err_skipped
        self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.minibatch_max_err_y_sum = self.ev.max_err_y_sum
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add gradient descent units
        self.gd = list(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(device=device)
        # self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        """
        for i in range(len(self.forward) - 2, -1, -1):
            if i:
                self.gd[i] = gd.GDTanh(device=device)
            else:
                self.gd[i] = gd.GDTanh(device=device)
                #self.gd[i].y_rand = self.forward[i].output_rand
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
        """
        self.rpt.link_from(self.gd[-1])

        self.end_point = units.EndPoint()
        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(2, 3):
            self.plt.append(plotters.SimplePlotter(figure_label="num errors",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if i == 2 else self.plt[-2])
            self.plt[-1].gate_skip = self.decision.epoch_ended
            self.plt[-1].gate_skip_not = [1]
        # self.plt[0].clear_plot = True
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(2, 3):
            self.plt_mx.append(plotters.MatrixPlotter(
                figure_label=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.plt[-1] if i == 2
                                      else self.plt_mx[-2])
            self.plt_mx[-1].gate_skip = self.decision.epoch_ended
            self.plt_mx[-1].gate_skip_not = [1]
        # err_y plotter
        self.plt_err_y = []
        for i in range(2, 3):
            self.plt_err_y.append(plotters.SimplePlotter(
                figure_label="Last layer max gradient sum",
                plot_style=styles[i]))
            self.plt_err_y[-1].input = self.decision.max_err_y_sums
            self.plt_err_y[-1].input_field = i
            self.plt_err_y[-1].link_from(self.plt_mx[-1] if i == 2
                                         else self.plt_err_y[-2])
            self.plt_err_y[-1].gate_skip = self.decision.epoch_ended
            self.plt_err_y[-1].gate_skip_not = [1]
        # self.plt_err_y[0].clear_plot = True
        self.gd[-1].link_from(self.plt_err_y[-1])

    def initialize(self):
        retval = self.start_point.initialize_dependent()
        if retval:
            return retval

    def run(self, threshold, threshold_low, global_alpha, global_lambda):
        self.ev.threshold = threshold
        self.ev.threshold_low = threshold_low
        for gd in self.gd:
            if gd == None:
                continue
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
        retval = self.start_point.run_dependent()
        if retval:
            return retval
        self.end_point.wait()


def main():
    # if __debug__:
    #    logging.basicConfig(level=logging.DEBUG)
    # else:
    logging.basicConfig(level=logging.INFO)
    """This is a test for correctness of a particular trained 2-layer network.
    fin = open("%s/mnist_rbm.pickle" % (config.snapshot_dir), "rb")
    w = pickle.load(fin)
    fin.close()

    weights = w.forward[0].weights.v
    i = 0
    for row in weights:
        img = row.reshape(28, 28).copy()
        img -= img.min()
        m = img.max()
        if m:
            img /= m
            img *= 255.0
        scipy.misc.imsave("/tmp/img/%03d.png" % (i), img.astype(numpy.uint8))
        i += 1

    logging.info("Done")
    sys.exit(0)
    """

    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    try:
        cl = opencl.DeviceList()
        device = cl.get_device()
        try:
            fin = open("%s/xor.pickle" % (config.snapshot_dir), "rb")
            w0 = pickle.load(fin)
            fin.close()
            layers = []
            for i in range(0, len(w0.forward) - 1):
                layers.append(w0.forward[i].output.batch.size //
                              w0.forward[i].output.batch.shape[0])
            layers.append(2)
            w = Workflow2(layers=layers, device=device)
            w.initialize()
            for i in range(0, len(w0.forward) - 1):
                w.forward[i].weights.v[:] = w0.forward[i].weights.v[:]
                w.forward[i].weights.update()
                w.forward[i].bias.v[:] = w0.forward[i].bias.v[:]
                w.forward[i].bias.update()
            w.run(threshold=1.0, threshold_low=1.0,
                  global_alpha=0.001, global_lambda=0.00005)
        except FileNotFoundError:
            w = Workflow(layers=[8, 3], device=device)
            w.initialize()
            w.run(threshold_ok=0.0005, threshold_skip=0.0,
                  global_alpha=0.001, global_lambda=0.00005)
    except KeyboardInterrupt:
        w.gd[-1].gate_block = [1]
    logging.info("Will snapshot in 15 seconds...")
    time.sleep(5)
    logging.info("Will snapshot in 10 seconds...")
    time.sleep(5)
    logging.info("Will snapshot in 5 seconds...")
    time.sleep(5)
    fnme = "%s/xor_stop.pickle" % (config.snapshot_dir)
    logging.info("Snapshotting to %s" % (fnme))
    fout = open(fnme, "wb")
    pickle.dump(w, fout)
    fout.close()

    plotters.Graphics().wait_finish()
    logging.debug("End of job")


if __name__ == "__main__":
    main()
    sys.exit()
