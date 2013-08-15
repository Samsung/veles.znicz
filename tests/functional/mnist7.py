#!/usr/bin/python3.3 -O
"""
Created on August 12, 2013

MNIST with target encoded as 7 points, MSE.

Such encoding is not working because of the addititional abstract layer.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import units
import formats
import struct
import error
import numpy
import config
import rnd
import opencl
import plotters
import pickle
import time
import logging
import glob


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
    def __init__(self, classes=[0, 10000, 60000], minibatch_max_size=60,
                 rnd=rnd.default):
        """Constructor.

        Parameters:
            classes: [test, validation, train],
                ints - in samples,
                floats - relative from (0 to 1).
            minibatch_size: minibatch max size.
        """
        super(Loader, self).__init__()
        self.rnd = [rnd]

        self.minibatch_data = formats.Vector()
        self.minibatch_indexes = formats.Vector()
        self.minibatch_labels = formats.Vector()
        self.minibatch_target = formats.Vector()

        self.minibatch_class = [0]
        self.minibatch_last = [0]

        self.total_samples = [70000]
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
                                                  self.total_samples))

        self.original_data = None
        self.original_labels = None
        self.original_target = None
        self.class_target = formats.Vector()

        self.shuffled_indexes = None

    def load_original(self, offs, labels_count, labels_fnme, images_fnme):
        """Loads data from original MNIST files.
        """
        self.log().info("Loading from original MNIST files...")

        # Reading labels:
        fin = open(labels_fnme, "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2049:
            raise error.ErrBadFormat("Wrong header in train-labels")

        n_labels, = struct.unpack(">i", fin.read(4))
        if n_labels != labels_count:
            raise error.ErrBadFormat("Wrong number of labels in train-labels")

        arr = numpy.fromfile(fin, dtype=numpy.byte, count=n_labels)
        if arr.size != n_labels:
            raise error.ErrBadFormat("EOF reached while reading labels from "
                                     "train-labels")
        self.original_labels[offs:offs + labels_count] = arr[:]
        if self.original_labels.min() != 0 or self.original_labels.max() != 9:
            raise error.ErrBadFormat("Wrong labels range in train-labels.")

        fin.close()

        # Reading images:
        fin = open(images_fnme, "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2051:
            raise error.ErrBadFormat("Wrong header in train-images")

        n_images, = struct.unpack(">i", fin.read(4))
        if n_images != n_labels:
            raise error.ErrBadFormat("Wrong number of images in train-images")

        n_rows, n_cols = struct.unpack(">2i", fin.read(8))
        if n_rows != 28 or n_cols != 28:
            raise error.ErrBadFormat("Wrong images size in train-images, "
                                     "should be 28*28")

        # 0 - white, 255 - black
        pixels = numpy.fromfile(fin, dtype=numpy.ubyte,
                                count=n_images * n_rows * n_cols)
        if pixels.shape[0] != n_images * n_rows * n_cols:
            raise error.ErrBadFormat("EOF reached while reading images "
                                     "from train-images")

        fin.close()

        # Transforming images into float arrays and normalizing to [-1, 1]:
        images = pixels.astype(config.dtypes[config.dtype]).\
            reshape(n_images, n_rows, n_cols)
        self.log().info("Original range: [%.1f, %.1f]" % (images.min(),
                                                images.max()))
        for image in images:
            normalize(image)
        self.log().info("Range after normalization: [%.1f, %.1f]" % (
                                            images.min(), images.max()))
        self.original_data[offs:offs + n_images] = images[:]
        self.log().info("Done")

    def initialize(self):
        """Here we will load MNIST data.
        """
        if not self.original_labels or self.original_labels.size < 70000:
            self.original_labels = numpy.zeros([70000], dtype=numpy.int8)
        if not self.original_data or self.original_data.shape[0] < 70000:
            self.original_data = \
            numpy.zeros([70000, 28, 28], dtype=config.dtypes[config.dtype])
        if not self.shuffled_indexes or self.shuffled_indexes.size < 70000:
            self.shuffled_indexes = numpy.arange(70000, dtype=numpy.int32)

        global this_dir
        self.load_original(0, 10000,
                           "%s/MNIST/t10k-labels.idx1-ubyte" % (this_dir),
                           "%s/MNIST/t10k-images.idx3-ubyte" % (this_dir))
        self.load_original(10000, 60000,
                           "%s/MNIST/train-labels.idx1-ubyte" % (this_dir),
                           "%s/MNIST/train-images.idx3-ubyte" % (this_dir))

        self.class_target.reset()
        self.class_target.v = numpy.array(
            [[1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],  # 0
             [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0],  # 1
             [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0],  # 2
             [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0],  # 3
             [-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0],  # 4
             [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],  # 5
             [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],  # 6
             [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0],  # 7
             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 8
             [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0]],  # 9)
            dtype=config.dtypes[config.dtype])
        self.original_target = numpy.zeros([self.original_labels.shape[0], 7],
            dtype=config.dtypes[config.dtype])
        for i in range(0, self.original_labels.shape[0]):
            label = self.original_labels[i]
            self.original_target[i] = self.class_target.v[label]

        sh = [self.minibatch_maxsize[0]]
        for i in self.original_data.shape[1:]:
            sh.append(i)
        self.minibatch_data.v = numpy.zeros(
            sh, dtype=config.dtypes[config.dtype])
        self.minibatch_labels.v = numpy.zeros(
            [self.minibatch_maxsize[0]], dtype=numpy.int8)
        self.minibatch_indexes.v = numpy.zeros(
            [self.minibatch_maxsize[0]], dtype=numpy.int32)
        sh = [self.minibatch_maxsize[0]]
        for i in self.original_target.shape[1:]:
            sh.append(i)
        self.minibatch_target.v = numpy.zeros(
            sh, dtype=config.dtypes[config.dtype])

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
        idxs = self.minibatch_indexes.v
        idxs[0:minibatch_size] = self.shuffled_indexes[self.minibatch_offs[0]:\
            self.minibatch_offs[0] + minibatch_size]

        self.minibatch_labels.v[0:minibatch_size] = \
            self.original_labels[idxs[0:minibatch_size]]

        self.minibatch_data.v[0:minibatch_size] = \
            self.original_data[idxs[0:minibatch_size]]

        self.minibatch_target.v[0:minibatch_size] = \
            self.original_target[idxs[0:minibatch_size]]

        # Fill excessive indexes.
        if minibatch_size < self.minibatch_maxsize[0]:
            self.minibatch_data.v[minibatch_size:] = 0.0
            self.minibatch_labels.v[minibatch_size:] = -1
            self.minibatch_indexes.v[minibatch_size:] = -1

        # Set update flag for GPU operation.
        self.minibatch_data.update()
        self.minibatch_labels.update()
        self.minibatch_indexes.update()
        self.minibatch_target.update()

        self.log().debug("%s in %.2f sec" % (self.__class__.__name__,
                                      time.time() - t1))


import all2all
import evaluator
import gd
import scipy.misc


class ImageSaverAE(units.Unit):
    """Saves input and output side by side to png for AutoEncoder.

    Attributes:
        out_dirs: output directories by minibatch_class where to save png.
        input: batch with input samples.
        output: batch with corresponding output samples.
        indexes: sample indexes.
        labels: sample labels.
    """
    def __init__(self, out_dirs=[".", ".", "."]):
        super(ImageSaverAE, self).__init__()
        self.out_dirs = out_dirs
        self.input = None  # formats.Vector()
        self.output = None  # formats.Vector()
        self.target = None  # formats.Vector()
        self.indexes = None  # formats.Vector()
        self.labels = None  # formats.Vector()
        self.minibatch_class = None  # [0]
        self.minibatch_size = None  # [0]
        self.this_save_date = [0]
        self.last_save_date = 0

    def run(self):
        self.input.sync()
        self.output.sync()
        self.target.sync()
        self.indexes.sync()
        self.labels.sync()
        if self.last_save_date < self.this_save_date[0]:
            self.last_save_date = self.this_save_date[0]
            for dirnme in self.out_dirs:
                try:
                    os.makedirs(dirnme)
                except OSError:
                    pass
                files = glob.glob("%s/*.png" % (dirnme))
                for file in files:
                    try:
                        os.unlink(file)
                    except OSError:
                        pass
        for i in range(0, self.minibatch_size[0]):
            x = self.input.v[i]
            y = self.output.v[i]
            t = self.target.v[i]
            idx = self.indexes.v[i]
            lbl = self.labels.v[i]
            mse = numpy.linalg.norm(t - y) / t.size
            fnme = "%s/%.6f_%d_%d.png" % (
                self.out_dirs[self.minibatch_class[0]], mse, lbl, idx)
            img = x.copy()
            img -= img.min()
            m = img.max()
            if m:
                img /= m
                img *= 255.0
            scipy.misc.imsave(fnme, img.astype(numpy.uint8))


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
    def __init__(self, fail_iterations=10000):
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
        self.threshold_ok = 0.0
        self.weights_to_sync = None

    def initialize(self):
        if (self.minibatch_metrics != None and
            self.minibatch_metrics.v != None):
            for i in range(0, len(self.epoch_metrics)):
                self.epoch_metrics[i] = (
                    numpy.zeros_like(self.minibatch_metrics.v))

    def run(self):
        if self.t1 == None:
            self.t1 = time.time()
        self.complete[0] = 0
        self.epoch_ended[0] = 0

        minibatch_class = self.minibatch_class[0]

        if self.minibatch_last[0]:
            self.minibatch_metrics.sync()
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
            if self.minibatch_class[0] == 1:
                if self.just_snapshotted[0]:
                    self.just_snapshotted[0] = 0
                    self.complete[0] = 1
                if self.epoch_min_mse[1] < self.min_validation_mse:
                    self.min_validation_mse = self.epoch_min_mse[1]
                    self.min_validation_mse_epoch_number = self.epoch_number[0]
                    if self.epoch_metrics[1][0] < self.threshold_ok:
                        if self.fnme != None:
                            try:
                                os.unlink(self.fnme)
                            except FileNotFoundError:
                                pass
                        self.fnme = "%s/mnist_ae.%.4f.pickle" % \
                            (config.snapshot_dir, self.epoch_metrics[1][0])
                        self.log().info("Snapshotting to %s" % (self.fnme))
                        fout = open(self.fnme, "wb")
                        pickle.dump(self.workflow, fout)
                        fout.close()
                        self.just_snapshotted[0] = 1
                # Stop condition
                if self.epoch_number[0] - \
                   self.min_validation_mse_epoch_number > \
                   self.fail_iterations[0]:
                    self.complete[0] = 1

            # Print some statistics
            t2 = time.time()
            self.log().info(
                "Epoch %d Class %d AvgMSE %.6f Err %d (%.2f%%) "
                "MaxMSE %.6f MinMSE %.2e in %.2f sec" % \
                (self.epoch_number[0], self.minibatch_class[0],
                 self.epoch_metrics[self.minibatch_class[0]][0],
                 self.n_err[self.minibatch_class[0]],
                 self.n_err_pt[self.minibatch_class[0]],
                 self.epoch_metrics[self.minibatch_class[0]][1],
                 self.epoch_metrics[self.minibatch_class[0]][2],
                 t2 - self.t1))
            self.t1 = t2

            # Training set processed
            if self.minibatch_class[0] == 2:
                #"""
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
                #"""
                self.epoch_ended[0] = 1
                self.epoch_number[0] += 1
                # Reset n_err
                for i in range(0, len(self.n_err)):
                    self.n_err[i] = 0
                # Sync weights
                if self.weights_to_sync != None:
                    self.weights_to_sync.sync()

            # Reset statistics per class
            self.minibatch_n_err.v[:] = 0
            self.minibatch_n_err.update()
            if (self.minibatch_metrics != None and
                self.minibatch_metrics.v != None):
                self.minibatch_metrics.v[:] = 0
                self.minibatch_metrics.v[2] = 1.0e30
                self.minibatch_metrics.update()


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
            #if not i:
            #    amp = None
            #else:
            #    amp = 9.0 / 1.7159 / layers[i - 1]
            amp = 0.1
            aa = all2all.All2AllTanh([layers[i]], device=device,
                                     weights_amplitude=amp)
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
        self.ev.target = self.loader.minibatch_target
        self.ev.class_target = self.loader.class_target
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = Decision()
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err_skipped
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add Image Saver unit
        self.image_saver = ImageSaverAE(["/tmp/img/test",
                                         "/tmp/img/validation",
                                         "/tmp/img/train"])
        self.image_saver.link_from(self.decision)
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.ev.y
        self.image_saver.target = self.ev.target
        self.image_saver.indexes = self.loader.minibatch_indexes
        self.image_saver.labels = self.loader.minibatch_labels
        self.image_saver.minibatch_class = self.loader.minibatch_class
        self.image_saver.minibatch_size = self.loader.minibatch_size
        self.image_saver.gate_skip = self.decision.just_snapshotted
        self.image_saver.gate_skip_not = [1]

        # Add gradient descent units
        self.gd = list(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDTanh(device=device)
        self.gd[-1].link_from(self.image_saver)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(device=device)
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
        for i in range(0, 3):
            self.plt.append(plotters.SimplePlotter(figure_label="mse",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else
                                   self.plt[-2])
            self.plt[-1].gate_block = (self.decision.epoch_ended if not i
                                       else [1])
            self.plt[-1].gate_block_not = [1]
        self.plt[0].clear_plot = True
        # Weights plotter
        #"""
        self.decision.weights_to_sync = self.gd[0].weights
        self.plt_mx = plotters.Weights2D(figure_label="First Layer Weights",
                                         limit=25)
        self.plt_mx.input = self.decision.weights_to_sync
        self.plt_mx.input_field = "v"
        self.plt_mx.get_shape_from = self.forward[0].input
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = self.decision.epoch_ended
        self.plt_mx.gate_block_not = [1]
        #"""
        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(0, 3):
            self.plt_max.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.decision if not i else
                                       self.plt_max[-2])
            self.plt_max[-1].gate_block = (self.decision.epoch_ended if not i
                                           else [1])
            self.plt_max[-1].gate_block_not = [1]
        self.plt_max[0].clear_plot = True
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(0, 3):
            self.plt_min.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.decision if not i else
                                       self.plt_min[-2])
            self.plt_min[-1].gate_block = (self.decision.epoch_ended if not i
                                           else [1])
            self.plt_min[-1].gate_block_not = [1]
        self.plt_min[0].clear_plot = True

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


def main():
    #if __debug__:
    #    logging.basicConfig(level=logging.DEBUG)
    #else:
    logging.basicConfig(level=logging.INFO)
    """This is a test for correctness of a particular trained 2-layer network.
    fin = open("mnist7.pickle", "rb")
    w = pickle.load(fin)
    fin.close()

    fout = open("Wb.pickle", "wb")
    pickle.dump((w.forward[0].weights.v, w.forward[0].bias.v), fout)
    fout.close()

    logging.info("Done")
    units.pool.shutdown()
    sys.exit(0)

    fout = open("w100.txt", "w")
    weights = w.forward[0].weights.v
    for row in weights:
        fout.write(" ".join("%.6f" % (x) for x in row))
        fout.write("\n")
    fout.close()
    fout = open("b100.txt", "w")
    bias = w.forward[0].bias.v
    fout.write(" ".join("%.6f" % (x) for x in bias))
    fout.write("\n")
    fout.close()

    a = w.loader.original_data.reshape(70000, 784)[0:10000]
    b = weights.transpose()
    c = numpy.zeros([10000, 100], dtype=a.dtype)
    numpy.dot(a, b, c)
    c[:] += bias
    c *= 0.6666
    numpy.tanh(c, c)
    c *= 1.7159

    fout = open("w10.txt", "w")
    weights = w.forward[1].weights.v
    for row in weights:
        fout.write(" ".join("%.6f" % (x) for x in row))
        fout.write("\n")
    fout.close()
    fout = open("b10.txt", "w")
    bias = w.forward[1].bias.v
    fout.write(" ".join("%.6f" % (x) for x in bias))
    fout.write("\n")
    fout.close()

    a = c
    b = weights.transpose()
    c = numpy.zeros([10000, 10], dtype=a.dtype)
    numpy.dot(a, b, c)
    c[:] += bias

    labels = w.loader.original_labels[0:10000]
    n_ok = 0
    for i in range(0, 10000):
        im = numpy.argmax(c[i])
        if im == labels[i]:
            n_ok += 1
    logging.info("%d errors" % (10000 - n_ok))

    logging.info("Done")
    units.pool.shutdown()
    sys.exit(0)
    """

    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    try:
        cl = opencl.DeviceList()
        device = cl.get_device()
        w = Workflow(layers=[100, 100, 7], device=device)
        w.initialize()
    except KeyboardInterrupt:
        return
    try:
        w.run(threshold_ok=0.00001, threshold_skip=0.0,
              global_alpha=0.0001, global_lambda=0.00005)
    except KeyboardInterrupt:
        w.gd[-1].gate_block = [1]
    logging.info("Will snapshot in 15 seconds...")
    time.sleep(5)
    logging.info("Will snapshot in 10 seconds...")
    time.sleep(5)
    logging.info("Will snapshot in 5 seconds...")
    time.sleep(5)
    fnme = "%s/mnist7.pickle" % (config.snapshot_dir)
    logging.info("Snapshotting to %s" % (fnme))
    fout = open(fnme, "wb")
    pickle.dump(w, fout)
    fout.close()

    plotters.Graphics().wait_finish()
    logging.info("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
