# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 12, 2014

Kohonen map demo on a simple two dimension dataset.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import numpy
from zope.interface import implementer

from veles.config import root
import veles.error as error
from veles.interaction import Shell
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.kohonen as kohonen
import veles.loader as loader


@implementer(loader.IFullBatchLoader)
class KohonenLoader(loader.FullBatchLoader):
    """Loads the sample dataset.
    """

    def load_data(self):
        """Here we will load MNIST data.
        """
        file_name = root.kohonen.loader.dataset_file
        try:
            data = numpy.loadtxt(file_name)
        except:
            raise error.BadFormatError("Could not load data from %s" %
                                       file_name)
        if data.shape != (2, 1000):
            raise error.BadFormatError("Data in %s has the invalid shape" %
                                       file_name)

        self.original_data.mem = numpy.zeros((1000, 2), dtype=self.dtype)
        self.original_data.mem[:, 0] = data[0]
        self.original_data.mem[:, 1] = data[1]

        self.class_lengths[0] = 0
        self.class_lengths[1] = 0
        self.class_lengths[2] = 1000


class KohonenWorkflow(nn_units.NNWorkflow):
    """Workflow for Kohonen sample dataset.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "Kohonen")
        super(KohonenWorkflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = KohonenLoader(
            self, name="Kohonen fullbatch loader",
            minibatch_size=root.kohonen.loader.minibatch_size,
            force_numpy=root.kohonen.loader.force_numpy)
        self.loader.link_from(self.repeater)

        # Kohonen training layer
        self.trainer = kohonen.KohonenTrainer(
            self, shape=root.kohonen.forward.shape,
            weights_filling=root.kohonen.forward.weights_filling,
            weights_stddev=root.kohonen.forward.weights_stddev,
            gradient_decay=root.kohonen.train.gradient_decay,
            radius_decay=root.kohonen.train.radius_decay)
        self.trainer.link_from(self.loader)
        self.trainer.link_attrs(self.loader, ("input", "minibatch_data"))

        # Loop decision
        self.decision = kohonen.KohonenDecision(
            self, max_epochs=root.kohonen.decision.epochs)
        self.decision.link_from(self.trainer)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "last_minibatch",
                                 "class_lengths",
                                 "epoch_ended",
                                 "epoch_number")
        self.decision.link_attrs(self.trainer, "weights", "winners")

        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.decision.epoch_ended

        self.repeater.link_from(self.ipython)
        self.ipython.gate_block = self.decision.complete

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plotters = [nn_plotting_units.KohonenHits(self),
                         nn_plotting_units.KohonenInputMaps(self),
                         nn_plotting_units.KohonenNeighborMap(self)]
        self.plotters[0].link_attrs(self.trainer, "shape") \
            .link_from(self.ipython)
        self.plotters[0].input = self.decision.winners_mem
        self.plotters[0].gate_block = ~self.decision.epoch_ended
        self.plotters[1].link_attrs(self.trainer, "shape") \
            .link_from(self.ipython)
        self.plotters[1].input = self.decision.weights_mem
        self.plotters[1].gate_block = ~self.decision.epoch_ended
        self.plotters[2].link_attrs(self.trainer, "shape") \
            .link_from(self.ipython)
        self.plotters[2].input = self.decision.weights_mem
        self.plotters[2].gate_block = ~self.decision.epoch_ended

    def initialize(self, device, **kwargs):
        return super(KohonenWorkflow, self).initialize(device=device, **kwargs)


def run(load, main):
    load(KohonenWorkflow)
    main()
