#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

Lee Man Ha spam all2all solution.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root
import veles.plotting_units as plotting_units
from veles.snapshotter import Snapshotter
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
from veles.interaction import Shell
from veles.znicz.tests.research.spam_kohonen import Loader


spam_dir = os.path.join(os.path.dirname(__file__), "spam")

root.defaults = {"decision": {"fail_iterations": 100,
                              "store_samples_mse": True},
                 "snapshotter": {"prefix": "spam"},
                 "loader": {"minibatch_maxsize": 60,
                            "file": os.path.join(spam_dir, "data.txt.xz"),
                            "validation_ratio": 0.15},
                 "spam": {"learning_rate": 0.001,
                          "weights_decay": 0.0,
                          "layers": [100, 2]}}


class Workflow(nn_units.NNWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    """
    def __init__(self, workflow, layers, **kwargs):
        kwargs["name"] = kwargs.get("name", "Spam")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, name="Spam fullbatch loader",
                             minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.fwds[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(
                    self, output_shape=[layers[i]])
            else:
                aa = all2all.All2AllSoftmax(
                    self, output_shape=[layers[i]])
            self.fwds.append(aa)
            if i:
                self.fwds[i].link_from(self.fwds[i - 1])
                self.fwds[i].link_attrs(self.fwds[i - 1],
                                        ("input", "output"))
            else:
                self.fwds[i].link_from(self.loader)
                self.fwds[i].link_attrs(self.loader,
                                        ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.fwds[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("labels", "minibatch_labels"),
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.DecisionGD(self)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(
            self.loader, "minibatch_class", "minibatch_size",
            "minibatch_offset", "no_more_minibatches_left", "class_samples")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_output_sum"))
        self.decision.fwds = self.fwds
        self.decision.gds = self.gds
        self.decision.evaluator = self.evaluator

        self.snapshotter = Snapshotter(self, prefix=root.snapshotter.prefix,
                                       directory=root.common.snapshot_dir)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_block = \
            (~self.decision.epoch_ended | ~self.decision.improved)

        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.decision.epoch_ended

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(list(None for i in range(0, len(self.fwds))))
        self.gds[-1] = gd.GDSM(
            self, learning_rate=root.spam.learning_rate,
            gradient_moment=root.spam.gradient_moment,
            gradient_moment_bias=root.spam.gradient_moment_bias)
        self.gds[-1].link_from(self.ipython)
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.fwds[-1],
                                ("output", "output"),
                                ("input", "input"),
                                "weights", "bias")
        self.gds[-1].gate_skip = self.decision.gd_skip
        self.gds[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(
                self, learning_rate=root.spam.learning_rate,
                gradient_moment=root.spam.gradient_moment,
                gradient_moment_bias=root.spam.gradient_moment_bias)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].link_attrs(self.fwds[i], "output", "input",
                                   "weights", "bias")
            self.gds[i].gate_skip = self.decision.gd_skip
            self.gds[i].link_attrs(self.loader,
                                   ("batch_size", "minibatch_size"))
        self.gds[0].need_err_input = False
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["g-", "r-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="Errors", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if i == 1 else self.plt[-2])
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True

    def initialize(self, learning_rate, weights_decay, device):
        return super(Workflow, self).initialize(learning_rate=learning_rate,
                                                weights_decay=weights_decay,
                                                device=device)


def run(load, main):
    load(Workflow, layers=root.spam.layers)
    main(learning_rate=root.spam.learning_rate,
         weights_decay=root.spam.weights_decay)
