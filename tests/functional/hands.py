#!/usr/bin/python3.3 -O
"""
Created on Jun 14, 2013

File for Hands dataset.

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
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import formats
import numpy
import config
import rnd
import opencl
import plotters
import hog
import loader
import decision


class Loader(loader.ImageLoader):
    """Loads Hands dataset.
    """
    def from_image(self, fnme):
        a = numpy.fromfile(fnme, dtype=numpy.uint8).astype(numpy.float32)
        sx = int(numpy.sqrt(a.size))
        a = hog.hog(a.reshape(sx, sx)).astype(numpy.float32)
        formats.normalize(a)
        return a

    def get_label_from_filename(self, filename):
        lbl = 1 if filename.find("Positive") >= 0 else 0
        return lbl


import all2all
import evaluator
import gd
import workflow


class Workflow(workflow.NNWorkflow):
    """Sample workflow for Hands dataset.
    """
    def __init__(self, layers=None, device=None):
        super(Workflow, self).__init__(device=device)

        self.rpt.link_from(self.start_point)

        self.loader = Loader(validation_paths=[
            "%s/hands/Positive/Testing/*.raw" % (config.test_dataset_root,),
            "%s/hands/Negative/Testing/*.raw" % (config.test_dataset_root,)],
                             train_paths=[
            "%s/hands/Positive/Training/*.raw" % (config.test_dataset_root,),
            "%s/hands/Negative/Training/*.raw" % (config.test_dataset_root,)])
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward.clear()
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh([layers[i]], device=device)
            else:
                aa = all2all.All2AllSoftmax([layers[i]], device=device)
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
        self.decision = decision.Decision(snapshot_prefix="hands")
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(device=device)
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

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotters.SimplePlotter(figure_label="num errors",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else self.plt[-2])
            self.plt[-1].gate_block = (self.decision.epoch_ended if not i
                                       else [1])
            self.plt[-1].gate_block_not = [1]
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(0, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotters.MatrixPlotter(
                figure_label=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = self.decision.epoch_ended
            self.plt_mx[-1].gate_block_not = [1]

    def initialize(self, global_alpha, global_lambda):
        for gd in self.gd:
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
        return self.start_point.initialize_recursively()


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir,),
                                    numpy.int32, 1024))
    #rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    cl = opencl.DeviceList()
    device = cl.get_device()
    w = Workflow(layers=[30, 2], device=device)
    w.initialize(global_alpha=0.05, global_lambda=0.0)
    w.run()

    plotters.Graphics().wait_finish()
    logging.debug("End of job")


if __name__ == "__main__":
    main()
    sys.exit()
