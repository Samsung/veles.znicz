#!/usr/bin/python3 -O
"""
Created on August 4, 2013

Model created for class of wine recognition. Database - Wine.
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root
from veles.znicz.nn_units import NNSnapshotter
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
from .loader_wine import WineLoader


root.common.plotters_disabled = True

root.wine.update({
    "decision": {"fail_iterations": 200, "max_epochs": 100},
    "snapshotter": {"prefix": "wine", "time_interval": 1},
    "loader": {"minibatch_size": 10, "on_device": True},
    "learning_rate": 0.3,
    "weights_decay": 0.0,
    "layers": [8, 3]})


class WineWorkflow(nn_units.NNWorkflow):
    """Model created for class of wine recognition. Database - Wine.
    Model - fully-connected Neural Network with SoftMax loss function.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(WineWorkflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = WineLoader(
            self, minibatch_size=root.wine.loader.minibatch_size,
            on_device=root.wine.loader.on_device)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.forwards[:]
        for i in range(len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                         device=device)
            else:
                aa = all2all.All2AllSoftmax(self, output_shape=[layers[i]],
                                            device=device)
            self.forwards.append(aa)
            if i:
                self.forwards[-1].link_from(self.forwards[-2])
                self.forwards[-1].link_attrs(
                    self.forwards[-2], ("input", "output"))
            else:
                self.forwards[-1].link_from(self.loader)
                self.forwards[-1].link_attrs(
                    self.loader, ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.forwards[-1])
        self.evaluator.link_attrs(self.forwards[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("labels", "minibatch_labels"))

        # Add decision unit
        self.decision = decision.DecisionGD(
            self, fail_iterations=root.wine.decision.fail_iterations,
            max_epochs=root.wine.decision.max_epochs)
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

        self.snapshotter = NNSnapshotter(
            self, prefix=root.wine.snapshotter.prefix,
            directory=root.common.snapshot_dir, compress="",
            time_interval=root.wine.snapshotter.time_interval)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.loader.epoch_ended | ~self.decision.improved)

        self.end_point.link_from(self.snapshotter)
        self.end_point.gate_block = ~self.decision.complete

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(None for i in range(0, len(self.forwards)))
        self.gds[-1] = gd.GDSM(self, device=device)
        self.gds[-1].link_from(self.snapshotter)
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.forwards[-1], "output", "input",
                                "weights", "bias")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.forwards) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].link_attrs(self.forwards[i], "output", "input",
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader,
                                   ("batch_size", "minibatch_size"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.gds[0].need_err_input = False
        self.repeater.link_from(self.gds[0])

        self.loader.gate_block = self.decision.complete
        self.gds[-1].gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device, **kwargs):
        super(WineWorkflow, self).initialize(learning_rate=learning_rate,
                                             weights_decay=weights_decay,
                                             device=device, **kwargs)


def run(load, main):
    load(WineWorkflow, layers=root.wine.layers)
    main(learning_rate=root.wine.learning_rate,
         weights_decay=root.wine.weights_decay)
