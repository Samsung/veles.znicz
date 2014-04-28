#!/usr/bin/python3.3 -O
"""
Created on Jun 14, 2013

File for Hands dataset.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os

from veles.config import root
import veles.formats as formats
import veles.external.hog as hog
from veles.mutable import Bool
import veles.plotting_units as plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader

train_dir = [os.path.join(root.common.test_dataset_root,
                          "hands/Positive/Training/*.raw"),
             os.path.join(root.common.test_dataset_root,
                          "hands/Negative/Training/*.raw")]
validation_dir = [os.path.join(root.common.test_dataset_root,
                               "hands/Positive/Testing/*.raw"),
                  os.path.join(root.common.test_dataset_root,
                               "hands/Negative/Testing/*.raw")]

root.defaults = {"decision": {"fail_iterations": 100,
                              "snapshot_prefix": "hands"},
                 "loader": {"minibatch_maxsize": 60},
                 "hands": {"global_alpha": 0.05,
                           "global_lambda": 0.0,
                           "layers": [30, 2],
                           "path_for_load_data": {"train": train_dir,
                                                  "validation":
                                                  validation_dir}}}


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


class Workflow(nn_units.NNWorkflow):
    """Sample workflow for Hands dataset.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(
            self, validation_paths=root.hands.path_for_load_data.validation,
            train_paths=root.hands.path_for_load_data.train,
            minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.fwds[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                         device=device)
            else:
                aa = all2all.All2AllSoftmax(self, output_shape=[layers[i]],
                                            device=device)
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
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.fwds[-1], ("y", "output"), "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("labels", "minibatch_labels"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.Decision(
            self, snapshot_prefix=root.decision.snapshot_prefix,
            fail_iterations=root.decision.fail_iterations)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "no_more_minibatches_left",
                                 "class_samples")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"))

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(None for i in range(0, len(self.fwds)))
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
            self.gds[i] = gd.GDTanh(self, device=device)
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

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(0, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].link_attrs(self.decision,
                                       ("input", "confusion_matrixes"))
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended

    def initialize(self, global_alpha, global_lambda, device):
        super(Workflow, self).initialize(global_alpha=global_alpha,
                                         global_lambda=global_lambda,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.hands.layers)
    main(global_alpha=root.hands.global_alpha,
         global_lambda=root.hands.global_lambda)
