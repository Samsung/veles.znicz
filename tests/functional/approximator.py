#!/usr/bin/python3.3 -O
"""
Created on June 29, 2013

File for function approximation.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import os
import logging


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s/../src" % (this_dir))
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import units
import numpy
import rnd
import opencl
import plotters
import loader
import decision
import all2all
import evaluator
import gd
import scipy.io
import formats
import workflow
import error
import config


class Loader(loader.ImageLoader):
    def load_original(self, fnme):
        a = scipy.io.loadmat(fnme)
        for key in a.keys():
            if key[0] != "_":
                a = a[key]
                break
        else:
            raise error.ErrBadFormat("Could not find variable to import "
                                     "in %s" % (fnme))
        aa = numpy.zeros(a.shape, dtype=config.dtypes[config.dtype])
        aa[:] = a[:]
        ll = []
        for i in range(0, len(aa)):
            ll.append(self.total_samples[0] + i)
        return (aa, ll)

    def load_data(self):
        super(Loader, self).load_data()
        if self.class_samples[1] == 0:
            n = self.class_samples[2] * 10 // 70
            self.class_samples[1] = n
            self.class_samples[2] -= n

    def initialize(self):
        super(Loader, self).initialize()
        self.shuffle_validation_train()
        self.log().info("data range: (%.6f, %.6f), "
                        "target range: (%.6f, %.6f)" % (
            self.original_data.min(), self.original_data.max(),
            self.original_target.min(), self.original_target.max()))
        # Normalization
        train_data = self.original_data[self.nextclass_offs[1]:
                                        self.nextclass_offs[2]]
        train_target = self.original_target[self.nextclass_offs[1]:
                                            self.nextclass_offs[2]]

        self.data_IMul, self.data_IAdd = formats.normalize_pointwise(
                                                            train_data)
        self.target_IMul, self.target_IAdd = formats.normalize_pointwise(
                                                            train_target)

        train_data *= self.data_IMul
        train_data += self.data_IAdd
        train_target *= self.target_IMul
        train_target += self.target_IAdd

        train_data = self.original_data[self.nextclass_offs[1]:
                                        self.nextclass_offs[2]]
        train_target = self.original_target[self.nextclass_offs[1]:
                                            self.nextclass_offs[2]]

        self.log().info("train data normed range: (%.6f, %.6f)" % (
            train_data.min(), train_data.max()))
        self.log().info("train target normed range: (%.6f, %.6f)" % (
            train_target.min(), train_target.max()))

        if self.class_samples[0]:
            test_data = self.original_data[:self.nextclass_offs[0]]
            formats.assert_addr(test_data, self.original_data)
            test_target = self.original_target[:self.nextclass_offs[0]]
            formats.assert_addr(test_target, self.original_target)

            test_data *= self.data_IMul
            test_data += self.data_IAdd
            test_target *= self.target_IMul
            test_target += self.target_IAdd

            self.log().info("test data normed range: (%.6f, %.6f)" % (
                test_data.min(), test_data.max()))
            self.log().info("test target normed range: (%.6f, %.6f)" % (
                test_target.min(), test_target.max()))

        if self.class_samples[1]:
            validation_data = self.original_data[self.nextclass_offs[0]:
                                                 self.nextclass_offs[1]]
            formats.assert_addr(validation_data, self.original_data)
            validation_target = self.original_target[self.nextclass_offs[0]:
                                                     self.nextclass_offs[1]]
            formats.assert_addr(validation_target, self.original_target)

            validation_data *= self.data_IMul
            validation_data += self.data_IAdd
            validation_target *= self.target_IMul
            validation_target += self.target_IAdd

            self.log().info("validation data normed range: (%.6f, %.6f)" % (
                validation_data.min(), validation_data.max()))
            self.log().info("validation target normed range: (%.6f, %.6f)" % (
                validation_target.min(), validation_target.max()))


class Workflow(workflow.NNWorkflow):
    """Sample workflow for MNIST dataset.

    Attributes:
        start_point: start point.
        rpt: repeater.
        loader: loader.
        forward: list of all-to-all forward units.
        ev: evaluator softmax.
        decision: Decision.
        gd: list of gradient descent units.
    """
    def __init__(self, layers=None, device=None):
        super(Workflow, self).__init__(device=device)
        self.start_point = units.Unit()

        self.rpt = units.Repeater()
        self.rpt.link_from(self.start_point)

        self.loader = Loader(
            train_paths=["/data/veles/approximator/all_dec_appertures.mat"],
            target_paths=["/data/veles/approximator/all_org_appertures.mat"])
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward = []
        for i in range(0, len(layers)):
            aa = all2all.All2AllTanh([layers[i]], device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorMSE(device=device, threshold_ok=0.005)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.target = self.loader.minibatch_target
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(store_samples_mse=True,
                                          snapshot_prefix="approximator")
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

        # Add gradient descent units
        self.gd = list(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDTanh(device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.ev.effective_batch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.ev.effective_batch_size
        self.rpt.link_from(self.gd[0])

        self.end_point = units.EndPoint()
        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # Average plotter
        self.plt = []
        styles = ["", "b-", "k-"]  # ["r-", "b-", "k-"]
        j = 0
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt.append(plotters.SimplePlotter(figure_label="mse",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.plt[-2] if j else self.decision)
            self.plt[-1].gate_block = [1] if j else self.decision.epoch_ended
            self.plt[-1].gate_block_not = [1]
            j += 1
        self.plt[0].clear_plot = True
        # Max plotter
        self.plt_max = []
        styles = ["", "b--", "k--"]  # ["r--", "b--", "k--"]
        j = 0
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_max.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.plt_max[-2] if j else self.plt[-1])
            j += 1
        # Min plotter
        self.plt_min = []
        styles = ["", "b:", "k:"]  # ["r:", "b:", "k:"]
        j = 0
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_min.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.plt_min[-2] if j
                                       else self.plt_max[-1])
            j += 1
        self.plt_min[-1].redraw_plot = True
        # Histogram plotter
        self.plt_hist = plotters.MSEHistogram(figure_label="Histogram")
        self.plt_hist.link_from(self.decision)
        self.plt_hist.mse = self.decision.epoch_samples_mse[2]
        self.plt_hist.gate_block = self.decision.epoch_ended
        self.plt_hist.gate_block_not = [1]

    def initialize(self, device, threshold_ok, threshold_skip,
                   global_alpha, global_lambda,
                   minibatch_maxsize):
        for gd in self.gd:
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
            gd.device = device
        for forward in self.forward:
            forward.device = device
        self.ev.device = device
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        self.decision.threshold_ok = threshold_ok
        self.ev.threshold_ok = threshold_ok
        self.ev.threshold_skip = threshold_skip
        retval = self.start_point.initialize_dependent()
        if retval:
            return retval


def main():
    #if __debug__:
    #    logging.basicConfig(level=logging.DEBUG)
    #else:
    logging.basicConfig(level=logging.INFO)

    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 524288))
    cl = opencl.DeviceList()
    device = cl.get_device()
    w = Workflow(layers=[54, 9], device=device)
    w.initialize(threshold_ok=0.005, threshold_skip=0.0,
                 global_alpha=0.005, global_lambda=0.00005,
                 minibatch_maxsize=54, device=device)
    w.run()

    plotters.Graphics().wait_finish()
    logging.info("End of job")


if __name__ == "__main__":
    main()
    sys.exit()
