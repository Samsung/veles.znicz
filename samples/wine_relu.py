#!/usr/bin/python3.3 -O
"""
Created on August 4, 2013

File for Wine dataset (NN with RELU activation).

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os

from veles.config import root
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader


root.common.defaults = {"plotters_disabled": True}

root.defaults = {"decision": {"fail_iterations": 250,
                              "snapshot_prefix": "wine_relu"},
                 "loader": {"minibatch_maxsize": 1000000},
                 "wine_relu": {"global_alpha": 0.75,
                               "global_lambda": 0.0,
                               "layers": [10, 3],
                               "path_for_load_data":
                               os.path.join(root.common.veles_dir,
                                            "veles/znicz/samples/wine/" +
                                            "wine.data")}}


class Loader(loader.FullBatchLoader):
    """Loads Wine dataset.
    """
    def load_data(self):
        fin = open(root.wine_relu.path_for_load_data, "r")
        aa = []
        max_lbl = 0
        while True:
            s = fin.readline()
            if not len(s):
                break
            aa.append(
                numpy.fromstring(s, sep=",",
                                 dtype=opencl_types.dtypes[root.common.dtype]))
            max_lbl = max(max_lbl, int(aa[-1][0]))
        fin.close()

        self.original_data = numpy.zeros([len(aa), aa[0].shape[0] - 1],
                                         dtype=numpy.float32)
        self.original_labels = numpy.zeros(
            [self.original_data.shape[0]],
            dtype=opencl_types.itypes[
                opencl_types.get_itype_from_size(max_lbl)])

        for i, a in enumerate(aa):
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

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self,
                             minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.fwds[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllRELU(self, output_shape=[layers[i]],
                                         device=device)
            else:
                aa = all2all.All2AllSoftmax(self, output_shape=[layers[i]],
                                            device=device)
            self.fwds.append(aa)
            if i:
                self.fwds[i].link_from(self.fwds[i - 1])
                self.fwds[i].input = self.fwds[i - 1].output
            else:
                self.fwds[i].link_from(self.loader)
                self.fwds[i].input = self.loader.minibatch_data

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
            self,
            snapshot_prefix=root.decision.snapshot_prefix,
            fail_iterations=root.decision.fail_iterations)
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
        self.gds.extend(None for i in range(0, len(self.fwds)))
        self.gds[-1] = gd.GDSM(self, device=device)
        # self.gds[-1].link_from(self.decision)
        self.gds[-1].link_attrs(self.fwds[-1],
                               ("y", "output"),
                               ("h", "input"),
                               "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_y")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDRELU(self, device=device)
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

        self.gds[-1].link_from(self.decision)

    def initialize(self, global_alpha, global_lambda, device):
        super(Workflow, self).initialize(global_alpha=global_alpha,
                                         global_lambda=global_lambda,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.wine_relu.layers)
    main(global_alpha=root.wine_relu.global_alpha,
         global_lambda=root.wine_relu.global_lambda)
