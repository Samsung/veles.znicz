#!/usr/bin/python3.3 -O
"""
Created on August 4, 2013

File for Wine dataset.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import numpy
import os

from veles.config import root
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.rnd as rnd
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader


root.common.defaults = {"plotters_disabled": True}

root.defaults = {"decision": {"fail_iterations": 200,
                              "snapshot_prefix": "wine"},
                 "loader": {"minibatch_maxsize": 1000000,
                            "rnd": rnd.default,
                            "view_group": "LOADER"},
                 "wine": {"global_alpha": 0.5,
                          "global_lambda": 0.0,
                          "layers": [8, 3],
                          "path_for_load_data":
                          os.path.join(root.common.veles_dir,
                                       "veles/znicz/samples/wine/wine.data")}}


class Loader(loader.FullBatchLoader):
    """Loads Wine dataset.
    """
    def load_data(self):
        fin = open(root.wine.path_for_load_data, "r")
        lines = []
        max_lbl = 0
        while True:
            s = fin.readline()
            if not len(s):
                break
            lines.append(numpy.fromstring(
                s, sep=",", dtype=opencl_types.dtypes[root.common.dtype]))
            max_lbl = max(max_lbl, int(lines[-1][0]))
        fin.close()

        self.original_data = numpy.zeros([len(lines), lines[0].shape[0] - 1],
                                         dtype=numpy.float32)
        self.original_labels = numpy.zeros(
            [self.original_data.shape[0]],
            dtype=opencl_types.itypes[
                opencl_types.get_itype_from_size(max_lbl)])

        for i, a in enumerate(lines):
            self.original_data[i] = a[1:]
            self.original_labels[i] = int(a[0]) - 1
            # formats.normalize(self.original_data[i])

        IMul, IAdd = formats.normalize_pointwise(self.original_data)
        self.original_data[:] *= IMul
        self.original_data[:] += IAdd

        self.class_samples[0] = 0
        self.class_samples[1] = 0
        self.class_samples[2] = self.original_data.shape[0]

        self.nextclass_offsets[0] = 0
        self.nextclass_offsets[1] = 0
        self.nextclass_offsets[2] = self.original_data.shape[0]

        self.total_samples = self.original_data.shape[0]


class Workflow(nn_units.NNWorkflow):
    """Sample workflow for Wine dataset.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.rpt.link_from(self.start_point)

        self.loader = Loader(self,
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
                self.forward[-1].link_from(self.forward[-2])
                self.forward[-1].link_attrs(self.forward[-2],
                                            ("input", "output"))
            else:
                self.forward[-1].link_from(self.loader)
                self.forward[-1].link_attrs(self.loader,
                                            ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(self, device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.link_attrs(self.forward[-1],
                           ("y", "output"),
                           "max_idx")
        self.ev.link_attrs(self.loader,
                           ("batch_size", "minibatch_size"),
                           ("max_samples_per_epoch", "total_samples"),
                           ("labels", "minibatch_labels"))

        # Add decision unit
        self.decision = decision.Decision(
            self,
            snapshot_prefix=root.decision.snapshot_prefix,
            fail_iterations=root.decision.fail_iterations)
        self.decision.link_from(self.ev)
        self.decision.link_attrs(self.loader, "minibatch_class",
                                 "no_more_minibatches_left", "class_samples")
        self.decision.link_attrs(
            self.ev,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_y_sum"))

        # Add gradient descent units
        del self.gd[:]
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(self, device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].link_attrs(self.ev, "err_y")
        self.gd[-1].link_attrs(self.forward[-1],
                               ("y", "output"),
                               ("h", "input"),
                               "weights", "bias")
        self.gd[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gd[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(self, device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].link_attrs(self.gd[i + 1],
                                  ("err_y", "err_h"))
            self.gd[i].link_attrs(self.forward[i],
                                  ("y", "output"),
                                  ("h", "input"),
                                  "weights", "bias")
            self.gd[i].link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"))
            self.gd[i].gate_skip = self.decision.gd_skip
        self.rpt.link_from(self.gd[0])

        self.end_point.link_from(self.gd[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

    def initialize(self, global_alpha, global_lambda, device):
        self.ev.device = device
        for g in self.gd:
            g.device = device
            g.global_alpha = global_alpha
            g.global_lambda = global_lambda
        for forward in self.forward:
            forward.device = device
        return super(Workflow, self).initialize()


def run(load, main):
    load(Workflow, layers=root.wine.layers)
    main(global_alpha=root.wine.global_alpha,
         global_lambda=root.wine.global_lambda)
