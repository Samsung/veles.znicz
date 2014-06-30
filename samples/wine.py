#!/usr/bin/python3.3 -O
"""
Created on August 4, 2013

File for Wine dataset.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
from zope.interface import implementer

from veles.config import root
import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.znicz.nn_units import NNSnapshotter
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader


root.common.defaults = {"plotters_disabled": True}

root.defaults = {"decision": {"fail_iterations": 200},
                 "snapshotter": {"prefix": "wine"},
                 "loader": {"minibatch_size": 10},
                 "wine": {"learning_rate": 0.3,
                          "weights_decay": 0.0,
                          "layers": [8, 3],
                          "data_paths":
                          os.path.join(root.common.veles_dir,
                                       "veles/znicz/samples/wine/wine.data")}}


@implementer(loader.IFullBatchLoader)
class Loader(loader.FullBatchLoader):
    """Loads Wine dataset.
    """
    def load_data(self):
        fin = open(root.wine.data_paths, "r")
        lines = []
        max_lbl = 0
        while True:
            s = fin.readline()
            if not len(s):
                break
            lines.append(numpy.fromstring(
                s, sep=",",
                dtype=opencl_types.dtypes[root.common.precision_type]))
            max_lbl = max(max_lbl, int(lines[-1][0]))
        fin.close()

        self.original_data = numpy.zeros([len(lines), lines[0].shape[0] - 1],
                                         dtype=numpy.float32)
        self.original_labels = numpy.zeros(self.original_data.shape[0],
                                           dtype=numpy.int32)

        for i, a in enumerate(lines):
            self.original_data[i] = a[1:]
            self.original_labels[i] = int(a[0]) - 1
            # formats.normalize(self.original_data[i])

        IMul, IAdd = formats.normalize_pointwise(self.original_data)
        self.original_data[:] *= IMul
        self.original_data[:] += IAdd

        self.class_lengths[0] = 0
        self.class_lengths[1] = 0
        self.class_lengths[2] = self.original_data.shape[0]


class Workflow(nn_units.NNWorkflow):
    """Sample workflow for Wine dataset.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self,
                             minibatch_size=root.loader.minibatch_size)
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
                self.fwds[-1].link_from(self.fwds[-2])
                self.fwds[-1].link_attrs(self.fwds[-2], ("input", "output"))
            else:
                self.fwds[-1].link_from(self.loader)
                self.fwds[-1].link_attrs(self.loader,
                                         ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.fwds[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("labels", "minibatch_labels"))

        # Add decision unit
        self.decision = decision.DecisionGD(
            self, fail_iterations=root.decision.fail_iterations)
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
            (~self.loader.epoch_ended | ~self.decision.improved)

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(None for i in range(0, len(self.fwds)))
        self.gds[-1] = gd.GDSM(self, device=device)
        self.gds[-1].link_from(self.snapshotter)
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.fwds[-1], "output", "input",
                                "weights", "bias")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].link_attrs(self.fwds[i], "output", "input",
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader,
                                   ("batch_size", "minibatch_size"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.gds[0].need_err_input = False
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device):
        super(Workflow, self).initialize(learning_rate=learning_rate,
                                         weights_decay=weights_decay,
                                         learning_rate_bias=learning_rate,
                                         weights_decay_bias=weights_decay,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.wine.layers)
    main(learning_rate=root.wine.learning_rate,
         weights_decay=root.wine.weights_decay)
