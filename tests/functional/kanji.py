#!/usr/bin/python3.3 -O
"""
Created on June 29, 2013

File for kanji recognition.

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


import numpy
import config
import rnd
import opencl
import plotters
import pickle
import loader
import decision
import image_saver
import all2all
import evaluator
import gd
import re


class Loader(loader.ImageLoader):
    """Loads dataset.

    Attributes:
        lbl_re_: regular expression for extracting label from filename.
    """
    def init_unpickled(self):
        super(Loader, self).init_unpickled()
        self.lbl_re_ = re.compile("/(\d+)\.[\w.-]+$")

    def get_label_from_filename(self, filename):
        res = self.lbl_re_.search(filename)
        if res == None:
            return
        lbl = int(res.group(1))
        return lbl


import workflow


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

        self.rpt.link_from(self.start_point)

        self.loader = Loader(train_paths=[
            "%s/kanji/train/*.png" % (config.test_dataset_root)],
                             target_paths=[
            "%s/kanji/target/*.png" % (config.test_dataset_root)])
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward.clear()
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
        self.ev = evaluator.EvaluatorMSE(device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.target = self.loader.minibatch_target
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(store_samples_mse=True,
                                          snapshot_prefix="kanji")
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.minibatch_mse = self.ev.mse
        self.decision.minibatch_offs = self.loader.minibatch_offs
        self.decision.minibatch_size = self.loader.minibatch_size
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDTanh(device=device)
        self.gd[-1].link_from(self.decision)
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

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["", "", "k-"]  # ["r-", "b-", "k-"]
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt.append(plotters.SimplePlotter(figure_label="mse",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision
                                   if len(self.plt) == 1
                                   else self.plt[-2])
            self.plt[-1].gate_block = (self.decision.epoch_ended
                                       if len(self.plt) == 1
                                       else [1])
            self.plt[-1].gate_block_not = [1]
        self.plt[0].clear_plot = True
        # Weights plotter
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_mx = plotters.Weights2D(figure_label="First Layer Weights")
        self.plt_mx.input = self.gd[0].weights
        self.plt_mx.input_field = "v"
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = self.decision.epoch_ended
        self.plt_mx.gate_block_not = [1]
        # Max plotter
        self.plt_max = []
        styles = ["", "", "k--"]  # ["r--", "b--", "k--"]
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_max.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.plt[-1]
                                       if len(self.plt_max) == 1
                                       else self.plt_max[-2])
        # Min plotter
        self.plt_min = []
        styles = ["", "", "k:"]  # ["r:", "b:", "k:"]
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_min.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.plt_max[-1]
                                       if len(self.plt_min) == 1
                                       else self.plt_min[-2])
        self.plt_min[-1].redraw_plot = True
        # Image plotter
        self.decision.vectors_to_sync[self.forward[0].input] = 1
        self.decision.vectors_to_sync[self.forward[-1].output] = 1
        self.decision.vectors_to_sync[self.ev.target] = 1
        self.plt_img = plotters.Image(figure_label="output sample")
        self.plt_img.inputs.append(self.decision)
        self.plt_img.input_fields.append("sample_input")
        self.plt_img.inputs.append(self.decision)
        self.plt_img.input_fields.append("sample_output")
        self.plt_img.inputs.append(self.decision)
        self.plt_img.input_fields.append("sample_target")
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_block = self.decision.epoch_ended
        self.plt_img.gate_block_not = [1]
        # Histogram plotter
        self.plt_hist = plotters.MSEHistogram(figure_label="Histogram")
        self.plt_hist.link_from(self.decision)
        self.plt_hist.mse = self.decision.epoch_samples_mse[2]
        self.plt_hist.gate_block = self.decision.epoch_ended
        self.plt_hist.gate_block_not = [1]

    def initialize(self, global_alpha, global_lambda, minibatch_maxsize,
                   device):
        for gd in self.gd:
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
            gd.device = device
        for forward in self.forward:
            forward.device = device
        self.ev.device = device
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        return self.start_point.initialize_recursively()

    def run(self, weights, bias):
        if weights != None:
            for i, forward in enumerate(self.forward):
                forward.weights.v[:] = weights[i][:]
                forward.weights.update()
        if bias != None:
            for i, forward in enumerate(self.forward):
                forward.bias.v[:] = bias[i][:]
                forward.bias.update()
        return super(Workflow, self).run()


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 524288))
    cl = opencl.DeviceList()
    device = cl.get_device()
    fnme = "%s/kanji.pickle" % (config.snapshot_dir)
    fin = None
    try:
        fin = open(fnme, "rb")
    except IOError:
        pass
    weights = None
    bias = None
    if fin != None:
        w = pickle.load(fin)
        fin.close()
        if type(w) == tuple:
            logging.info("Will load weights")
            weights = w[0]
            bias = w[1]
            fin = None
        else:
            logging.info("Will load workflow")
            logging.info("Weights and bias ranges per layer are:")
            for forward in w.forward:
                logging.info("%f %f %f %f" % (
                    forward.weights.v.min(), forward.weights.v.max(),
                    forward.bias.v.min(), forward.bias.v.max()))
            w.decision.just_snapshotted[0] = 1
    if fin == None:
        w = Workflow(layers=[3969, 3481, 24 * 24], device=device)
    w.initialize(global_alpha=0.001, global_lambda=0.00005,
                 minibatch_maxsize=1485, device=device)
    w.run(weights=weights, bias=bias)

    plotters.Graphics().wait_finish()
    logging.info("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
