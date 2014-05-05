#!/usr/bin/python3.3 -O
"""
Created on November 25, 2013

MNIST with Convolutional layer.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root
import veles.error as error
import veles.plotting_units as plotting_units
from veles.znicz.samples import mnist
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.conv as conv
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.gd_conv as gd_conv
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.pooling as pooling


root.defaults = {"decision": {"fail_iterations": 100,
                              "snapshot_prefix": "mnist_conv"},
                 "loader": {"minibatch_maxsize": 540},
                 "weights_plotter": {"limit": 64},
                 "mnist_conv": {"global_alpha": 0.005,
                                "global_lambda": 0.00005,
                                "layers":
                                [{"type": "conv", "n_kernels": 25,
                                  "kx": 9, "ky": 9}, 100, 10]}}


class Workflow(nn_units.NNWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    A deep learning method (advanced convolutional neural network) is used.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        kwargs["name"] = "Convolutional MNIST"
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = mnist.Loader(self)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.fwds[:]
        for i in range(0, len(layers)):
            layer = layers[i]
            if type(layer) == int:
                if i == len(layers) - 1:
                    aa = all2all.All2AllSoftmax(self, output_shape=[layer],
                                                device=device)
                else:
                    aa = all2all.All2AllTanh(self, output_shape=[layer],
                                             device=device)
            elif type(layer) == dict:
                if layer["type"] == "conv":
                    aa = conv.ConvTanh(
                        self, n_kernels=layer["n_kernels"],
                        kx=layer["kx"], ky=layer["ky"], device=device)
                elif layer["type"] == "max_pooling":
                    aa = pooling.MaxPooling(
                        self, kx=layer["kx"], ky=layer["ky"], device=device)
                elif layer["type"] == "avg_pooling":
                    aa = pooling.AvgPooling(
                        self, kx=layer["kx"], ky=layer["ky"], device=device)
                else:
                    raise error.ErrBadFormat(
                        "Unsupported layer type %s" % (layer["type"]))
            else:
                raise error.ErrBadFormat(
                    "layers element type should be int "
                    "for all-to-all or dictionary for "
                    "convolutional or pooling")
            self.fwds.append(aa)
            if i:
                self.fwds[i].link_from(self.fwds[i - 1])
                self.fwds[i].link_attrs(self.fwds[i - 1], ("input", "output"))
            else:
                self.fwds[i].link_from(self.loader)
                self.fwds[i].link_attrs(self.loader,
                                        ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.fwds[-1], ("y", "output"), "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("labels", "minibatch_labels"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.Decision(
            self, fail_iterations=root.decision.fail_iterations,
            snapshot_prefix=root.decision.snapshot_prefix)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "no_more_minibatches_left",
                                 "class_samples")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_y_sum"))

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(list(None for i in range(0, len(self.fwds))))
        self.gds[-1] = gd.GDSM(self, device=device)
        self.gds[-1].link_from(self.decision)
        self.gds[-1].link_attrs(self.fwds[-1],
                                ("y", "output"),
                                ("h", "input"),
                                "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_y")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.fwds) - 2, -1, -1):
            if isinstance(self.fwds[i], conv.Conv):
                obj = gd_conv.GDTanhConv(
                    self, n_kernels=self.fwds[i].n_kernels,
                    kx=self.fwds[i].kx, ky=self.fwds[i].ky,
                    device=device)
            elif isinstance(self.fwds[i], pooling.MaxPooling):
                obj = gd_pooling.GDMaxPooling(
                    self, kx=self.fwds[i].kx, ky=self.fwds[i].ky,
                    device=device)
                obj.h_offs = self.fwds[i].input_offs
            elif isinstance(self.fwds[i], pooling.AvgPooling):
                obj = gd_pooling.GDAvgPooling(
                    self, kx=self.fwds[i].kx, ky=self.fwds[i].ky,
                    device=device)
            else:
                obj = gd.GDTanh(self, device=device)
            self.gds[i] = obj
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.fwds[i],
                                   ("y", "output"),
                                   ("h", "input"),
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader, ("batch_size",
                                                 "minibatch_size"))
            self.gds[i].link_attrs(self.gds[i + 1], ("err_y", "err_h"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if len(self.plt) < 2
                                   else self.plt[-2])
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(1, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].link_attrs(self.decision,
                                       ("input", "confusion_matrixes"))
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended
        # err_y plotter
        self.plt_err_y = []
        for i in range(1, 3):
            self.plt_err_y.append(plotting_units.AccumulatingPlotter(
                self, name="Last layer max gradient sum",
                plot_style=styles[i]))
            self.plt_err_y[-1].link_attrs(self.decision,
                                          ("input", "max_err_y_sums"))
            self.plt_err_y[-1].input_field = i
            self.plt_err_y[-1].link_from(self.decision)
            self.plt_err_y[-1].gate_block = ~self.decision.epoch_ended
        self.plt_err_y[0].clear_plot = True
        self.plt_err_y[-1].redraw_plot = True
        # Weights plotter
        self.decision.vectors_to_sync[self.gds[0].weights] = 1
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit)
        self.plt_mx.link_attrs(self.gds[0], ("input", "weights"))
        self.plt_mx.input_field = "v"
        self.plt_mx.get_shape_from = (
            [self.fwds[0].kx, self.fwds[0].ky]
            if isinstance(self.fwds[0], conv.Conv)
            else self.fwds[0].input)
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended

    def initialize(self, global_alpha, global_lambda, minibatch_maxsize,
                   device):
        super(Workflow, self).initialize(global_alpha=global_alpha,
                                         global_lambda=global_lambda,
                                         minibatch_maxsize=minibatch_maxsize,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.mnist_conv.layers)
    """
    W = []
    b = []
    for f in w.fwds:
        W.append(f.weights.v)
        b.append(f.bias.v)
    fout = open("/tmp/Wb.pickle", "wb")
    pickle.dump((W, b), fout)
    fout.close()
    sys.exit(0)
    """
    # w = Workflow(None, layers=[
    #                     {"type": "conv", "n_kernels": 25, "kx": 9, "ky": 9},
    #                     {"type": "avg_pooling", "kx": 2, "ky": 2},  # 0.98%
    #                     100, 10], device=device)
    # w = Workflow(None, layers=[
    #                     {"type": "conv", "n_kernels": 50, "kx": 9, "ky": 9},
    #                     {"type": "avg_pooling", "kx": 2, "ky": 2},  # 10
    #                     {"type": "conv", "n_kernels": 200, "kx": 3, "ky": 3},
    #                     {"type": "avg_pooling", "kx": 2, "ky": 2},  # 4
    #                     100, 10], device=device)
    main(global_alpha=root.mnist_conv.global_alpha,
         global_lambda=root.mnist_conv.global_lambda,
         minibatch_maxsize=root.loader.minibatch_maxsize)
