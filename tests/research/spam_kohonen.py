#!/usr/bin/python3.3 -O
"""
Created on May 12, 2014

Kohonen Spam detection.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import numpy
import os

from veles.config import root
from veles.interaction import Shell
import veles.units as units
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.kohonen as kohonen
from veles.znicz.tests.research.spam import Loader


spam_dir = os.path.join(os.path.dirname(__file__), "spam")

root.defaults = {
    "forward": {"shape": (8, 8),
                "weights_stddev": 0.05,
                "weights_filling": "uniform"},
    "decision": {"snapshot_prefix": "spam_kohonen",
                 "epochs": 5},
    "loader": {"minibatch_maxsize": 60,
               "file": os.path.join(spam_dir, "data.txt.xz"),
               "validation_ratio": 0},
    "train": {"gradient_decay": lambda t: 0.001 / (1.0 + t * 0.00001),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.00001)},
    "exporter": {"file": "weights.txt"}}


class WeightsExporter(units.Unit):
    def __init__(self, workflow, file_name, **kwargs):
        super(WeightsExporter, self).__init__(workflow, **kwargs)
        self.weights = None
        self.file_name = file_name

    def run(self):
        numpy.savetxt(self.file_name, self.weights.mem)
        self.info("Exported the resulting weights to %s", self.file_name)


class Workflow(nn_units.NNWorkflow):
    """Workflow for Kohonen Spam Detection.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "Kohonen Spam")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, name="Kohonen Spam fullbatch loader",
                             minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.repeater)

        # Kohonen training layer
        self.trainer = kohonen.KohonenTrainer(
            self,
            shape=root.forward.shape,
            weights_filling=root.forward.weights_filling,
            weights_stddev=root.forward.weights_stddev,
            gradient_decay=root.train.gradient_decay,
            radius_decay=root.train.radius_decay)
        self.trainer.link_from(self.loader)
        self.trainer.link_attrs(self.loader, ("input", "minibatch_data"))

        # Loop decision
        self.decision = kohonen.KohonenDecision(
            self, max_epochs=root.decision.epochs)
        self.decision.link_from(self.trainer)
        self.decision.link_attrs(self.loader, "minibatch_class",
                                              "no_more_minibatches_left",
                                              "class_samples")
        self.decision.link_attrs(self.trainer, "weights", "winners")
        self.trainer.epoch_ended = self.decision.epoch_ended

        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.decision.epoch_ended

        self.repeater.link_from(self.ipython)

        self.exporter = WeightsExporter(self, root.exporter.file)
        self.exporter.link_from(self.decision)
        self.exporter.weights = self.trainer.weights
        self.exporter.gate_block = ~self.decision.complete

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plotters = [nn_plotting_units.KohonenHits(self),
                         nn_plotting_units.KohonenNeighborMap(self)]
        self.plotters[0].link_attrs(self.trainer, "shape")
        self.plotters[0].input = self.decision.winners_copy
        self.plotters[0].link_from(self.decision)
        self.plotters[0].gate_block = ~self.decision.epoch_ended
        self.plotters[1].link_attrs(self.trainer, "shape")
        self.plotters[1].input = self.decision.weights_copy
        self.plotters[1].link_from(self.decision)
        self.plotters[1].gate_block = ~self.decision.epoch_ended

    def initialize(self, device):
        return super(Workflow, self).initialize(device=device)


def run(load, main):
    load(Workflow)
    main()
