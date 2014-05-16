#!/usr/bin/python3.3 -O
"""
Created on May 12, 2014

Kohonen map demo on a simple two dimension dataset.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import numpy
import os

from veles.config import root
import veles.error as error
from veles.interaction import Shell
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.kohonen as kohonen
import veles.znicz.loader as loader


data_path = os.path.join(os.path.dirname(__file__), "kohonen")


root.defaults = {
    "forward": {"shape": (8, 8),
                "weights_stddev": 0.05,
                "weights_filling": "uniform"},
    "decision": {"snapshot_prefix": "kohonen",
                 "epochs": 200},
    "loader": {"minibatch_maxsize": 10,
               "dataset_file": os.path.join(data_path, "kohonen.txt")},
    "train": {"gradient_decay": lambda t: 0.1 / (1.0 + t * 0.05),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.05)}}


class Loader(loader.FullBatchLoader):
    """Loads the sample dataset.
    """

    def load_data(self):
        """Here we will load MNIST data.
        """
        file_name = root.loader.dataset_file
        try:
            data = numpy.loadtxt(file_name)
        except:
            raise error.ErrBadFormat("Could not load data from %s" % file_name)
        if data.shape != (2, 1000):
            raise error.ErrBadFormat("Data in %s has the invalid shape" %
                                     file_name)

        self.original_labels = None
        self.original_data = numpy.zeros((1000, 2), dtype=numpy.float32)
        self.original_data[:, 0] = data[0]
        self.original_data[:, 1] = data[1]

        self.class_samples[0] = 0
        self.class_samples[1] = 0
        self.class_samples[2] = 1000


class Workflow(nn_units.NNWorkflow):
    """Workflow for Kohonen sample dataset.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "Kohonen")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, name="Kohonen fullbatch loader",
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
        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.decision.epoch_ended

        self.repeater.link_from(self.ipython)

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plotters = [nn_plotting_units.KohonenHits(self),
                         nn_plotting_units.KohonenInputMaps(self),
                         nn_plotting_units.KohonenNeighborMap(self)]
        self.plotters[0].link_attrs(self.trainer, ("input", "winners"),
                                    "shape")
        self.plotters[0].link_from(self.decision)
        self.plotters[0].gate_block = ~self.decision.epoch_ended
        self.plotters[2].link_attrs(self.trainer, ("input", "weights"),
                                    "shape")
        self.plotters[2].link_from(self.decision)
        self.plotters[2].gate_block = ~self.decision.epoch_ended

    def initialize(self, device):
        return super(Workflow, self).initialize(device=device)


def run(load, main):
    load(Workflow)
    main()
