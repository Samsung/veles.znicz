#!/usr/bin/python3.3 -O
"""
Created on Jun 14, 2013

File for Hands dataset.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy
import os

from veles.config import root, get
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


root.update = {"decision": {"fail_iterations":
                            get(root.decision.fail_iterations, 100),
                            "snapshot_prefix":
                            get(root.decision.snapshot_prefix,
                                       "hands")},
               "global_alpha": get(root.global_alpha, 0.05),
               "global_lambda": get(root.global_lambda, 0.0),
               "layers_hands": get(root.layers_hands, [30, 2]),
               "loader": {"minibatch_maxsize":
                          get(root.loader.minibatch_maxsize, 60)},
               "path_for_train_data":
               get(root.path_for_train_data,
                          [os.path.join(root.common.test_dataset_root,
                                        "hands/Positive/Training/*.raw"),
                           os.path.join(root.common.test_dataset_root,
                                        "hands/Negative/Training/*.raw")]),
               "path_for_valid_data":
               get(root.path_for_valid_data,
                          [os.path.join(root.common.test_dataset_root,
                                        "hands/Positive/Testing/*.raw"),
                           os.path.join(root.common.test_dataset_root,
                                        "hands/Negative/Testing/*.raw")])}


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

        self.rpt.link_from(self.start_point)

        self.loader = Loader(
            self, validation_paths=root.path_for_valid_data,
            train_paths=root.path_for_train_data,
            minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.rpt)

        # Add forward units
        del self.forward[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                         device=device)
            else:
                aa = all2all.All2AllSoftmax(self, output_shape=[layers[i]],
                                            device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(self, device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.link_attrs(self.forward[-1], ("y", "output"), "max_idx")
        self.ev.link_attrs(self.loader,
                           ("batch_size", "minibatch_size"),
                           ("labels", "minibatch_labels"),
                           ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.Decision(
            self, snapshot_prefix=root.decision.snapshot_prefix,
            fail_iterations=root.decision.fail_iterations)
        self.decision.link_from(self.ev)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "minibatch_last",
                                 "class_samples")
        self.decision.link_attrs(
            self.ev,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"))

        # Add gradient descent units
        del self.gd[:]
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(self, device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].link_attrs(self.forward[-1],
                               ("y", "output"),
                               ("h", "input"),
                               "weights", "bias")
        self.gd[-1].link_attrs(self.ev, "err_y")
        self.gd[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gd[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(self, device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].link_attrs(self.forward[i],
                                  ("y", "output"),
                                  ("h", "input"),
                                  "weights", "bias")
            self.gd[i].link_attrs(self.loader, ("batch_size",
                                                "minibatch_size"))
            self.gd[i].link_attrs(self.gd[i + 1], ("err_y", "err_h"))
            self.gd[i].gate_skip = self.decision.gd_skip
        self.rpt.link_from(self.gd[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_n_err_pt
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
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended


def run(load, main):
    load(Workflow, layers=root.layers_hands)
    main()
