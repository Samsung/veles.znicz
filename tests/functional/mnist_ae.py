#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

File for MNIST dataset.

AutoEncoder version.

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
add_path("%s" % (this_dir))
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import units
import numpy
import config
import rnd
import opencl
import plotters
import pickle
import time
import logging
import glob
import mnist
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
        self.indexes = None  # formats.Vector()
        self.labels = None  # formats.Vector()
        self.minibatch_class = None  # [0]
        self.minibatch_size = None  # [0]
        self.this_save_date = [0]
        self.last_save_date = 0

    def run(self):
        self.input.sync()
        self.output.sync()
        self.indexes.sync()
        self.labels.sync()
        xy = None
        if self.last_save_date < self.this_save_date[0]:
            self.last_save_date = self.this_save_date[0]
            for dirnme in self.out_dirs:
                files = glob.glob("%s/*.png" % (dirnme))
                for file in files:
                    try:
                        os.unlink(file)
                    except OSError:
                        pass
        for i in range(0, self.minibatch_size[0]):
            x = self.input.v[i]
            y = self.output.v[i].reshape(x.shape)
            idx = self.indexes.v[i]
            lbl = self.labels.v[i]
            mse = numpy.linalg.norm(x - y) / x.size
            if xy == None:
                xy = numpy.empty([x.shape[0], x.shape[1] * 2], dtype=x.dtype)
            xy[:, :x.shape[1]] = x[:, :]
            xy[:, x.shape[1]:] = y[:, :]
            x = xy[:, :x.shape[1]]
            y = xy[:, x.shape[1]:]
            x *= -1.0
            x += 1.0
            x *= 127.5
            numpy.clip(x, 0, 255, x)
            y *= -1.0
            y += 1.0
            y *= 127.5
            numpy.clip(y, 0, 255, y)
            fnme = "%s/%.6f_%d_%d.png" % (
                self.out_dirs[self.minibatch_class[0]], mse, lbl, idx)
            scipy.misc.imsave(fnme, xy.astype(numpy.uint8))


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
        # self.prev_train_err = 1.0e30
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
                """
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
                """
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

        self.loader = mnist.Loader()
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward = []
        for i in range(0, len(layers)):
            if not i:
                amp = 9.0 / 784
            else:
                amp = 9.0 / 1.7159 / layers[i - 1]
            # amp = 0.05
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
        self.ev.target = self.loader.minibatch_data
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
        self.image_saver = ImageSaverAE(["img/test", "img/validation",
                                         "img/train"])
        self.image_saver.link_from(self.decision)
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.forward[-1].output
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
                                         limit=16)
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

    def initialize(self, threshold_ok, threshold_skip,
              global_alpha, global_lambda):
        self.ev.threshold_ok = threshold_ok
        self.ev.threshold_skip = threshold_skip
        self.decision.threshold_ok = threshold_ok
        for gd in self.gd:
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
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
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    """This is a test for correctness of a particular trained 2-layer network.
    fin = open("mnist_ae.pickle", "rb")
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
        w = Workflow(layers=[500, 784], device=device)
        w.initialize(threshold_ok=0.0005, threshold_skip=0.0,
              global_alpha=0.001, global_lambda=0.0001)
    except KeyboardInterrupt:
        return
    try:
        w.run(threshold_ok=0.0005, threshold_skip=0.0,
              global_alpha=0.001, global_lambda=0.0001)
    except KeyboardInterrupt:
        w.gd[-1].gate_block = [1]
    logging.info("Will snapshot in 15 seconds...")
    time.sleep(5)
    logging.info("Will snapshot in 10 seconds...")
    time.sleep(5)
    logging.info("Will snapshot in 5 seconds...")
    time.sleep(5)
    fnme = "%s/mnist_ae.pickle" % (config.snapshot_dir)
    logging.info("Snapshotting to %s" % (fnme))
    fout = open(fnme, "wb")
    pickle.dump(w, fout)
    fout.close()

    plotters.Graphics().wait_finish()
    logging.info("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
