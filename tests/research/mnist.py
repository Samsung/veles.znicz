#!/usr/bin/python3 -O
"""
Created on Mar 20, 2013

File for MNIST dataset (NN with RELU activation).

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root
import veles.plotting_units as plotting_units
# import veles.znicz.nn_plotting_units as nn_plotting_units
# import veles.znicz.conv as conv
# import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.lr_adjust as lra
from veles.znicz.samples.mnist import Loader
from veles.znicz.nn_units import NNSnapshotter
from veles.znicz.standard_workflow import StandardWorkflow

mnist_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "samples/MNIST")
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")

root.defaults = {
    "learning_rate_adjust": {"do": False},
    "decision": {"fail_iterations": 100,
                 "max_epochs": 1000000000},
    "snapshotter": {"prefix": "mnist"},
    "loader": {"minibatch_size": 60},
    "weights_plotter": {"limit": 64},
    "mnist": {"learning_rate": 0.03,
              "weights_decay": 0.0,
              "layers":
              [{"type": "all2all", "output_shape": 100,
                "learning_rate": 0.03, "weights_decay": 0.0,
                "gradient_moment": 0.0, "factor_ortho": 0.001,
                "weights_filling": "uniform", "weights_stddev": 0.05,
                "bias_filling": "uniform", "bias_stddev": 0.05},
               {"type": "activation_tanh"},
               {"type": "softmax", "output_shape": 10,
                "learning_rate": 0.03, "weights_decay": 0.0,
                "gradient_moment": 0.0, "factor_ortho": 0.0,
                "weights_filling": "uniform", "weights_stddev": 0.05,
                "bias_filling": "uniform", "bias_stddev": 0.05}],
              "data_paths": {"test_images":  test_image_dir,
                             "test_label": test_label_dir,
                             "train_images": train_image_dir,
                             "train_label": train_label_dir}}}


class Workflow(StandardWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        kwargs["name"] = kwargs.get("name", "MNIST")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, name="Mnist fullbatch loader",
                             minibatch_size=root.loader.minibatch_size,
                             on_device=True)
        self.loader.link_from(self.repeater)

        # Add fwds units
        self.parse_forwards_from_config()

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.fwds[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("labels", "minibatch_labels"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.DecisionGD(
            self, fail_iterations=root.decision.fail_iterations,
            max_epochs=root.decision.max_epochs)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class", "minibatch_size",
                                 "last_minibatch", "class_lengths",
                                 "epoch_ended", "epoch_number")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_output_sum"))

        self.snapshotter = NNSnapshotter(self, prefix=root.snapshotter.prefix,
                                         directory=root.common.snapshot_dir,
                                         compress="", time_interval=0)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.decision.epoch_ended | ~self.decision.improved)

        # Add gradient descent units
        self.create_gd_units_by_config()

        if root.learning_rate_adjust.do:
            # Add learning_rate_adjust unit

            self.learning_rate_adjust = lra.LearningRateAdjust(
                self, lr_function=lra.inv_adjust_policy(
                    0.01, 0.0001, 0.75))
            self.learning_rate_adjust.link_from(self.gds[0])
            self.learning_rate_adjust.add_gd_units(self.gds)

            self.repeater.link_from(self.learning_rate_adjust)
            self.end_point.link_from(self.learning_rate_adjust)
        else:
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
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Confusion matrix plotter
        """
        self.plt_mx = []
        for i in range(1, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].link_attrs(self.decision,
                                       ("input", "confusion_matrixes"))
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended
        """
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
        """
        # Weights plotter
        self.plt_mx = []
        prev_channels = 1
        for i in range(len(layers)):
            if (not isinstance(self.fwds[i], conv.Conv) and
                    not isinstance(self.fwds[i], all2all.All2All)):
                continue
            nme = "%s %s" % (i + 1, layers[i]["type"])
            self.info("Added: %s", nme)
            plt_mx = nn_plotting_units.Weights2D(
                self, name=nme, limit=root.weights_plotter.limit)
            self.plt_mx.append(plt_mx)
            self.plt_mx[-1].link_attrs(self.gds[i], ("input",
                                                     "gradient_weights"))
            self.plt_mx[-1].input_field = "mem"
            if isinstance(self.fwds[i], conv.Conv):
                self.plt_mx[-1].get_shape_from = (
                    [self.fwds[i].kx, self.fwds[i].ky, prev_channels])
                prev_channels = self.fwds[i].n_kernels
            if (layers[i].get("output_shape") is not None and
                    layers[i]["type"] != "softmax"):
                self.plt_mx[-1].link_attrs(self.fwds[i],
                                           ("get_shape_from", "input"))
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended
        """

    def initialize(self, device, learning_rate, weights_decay, **kwargs):
        super(Workflow, self).initialize(learning_rate=learning_rate,
                                         weights_decay=weights_decay,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.mnist.layers)
    main(learning_rate=root.mnist.learning_rate,
         weights_decay=root.mnist.weights_decay)
