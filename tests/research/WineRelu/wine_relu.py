# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on August 4, 2013

Model created for class of wine recognition. Database - Wine.
Model - fully-connected Neural Network with SoftMax loss function with RELU
activation.

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


from veles.config import root
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
from .loader_wine import WineLoader
from veles.znicz.nn_units import NNSnapshotter


root.common.disable_plotting = True

root.wine_relu.update({
    "decision": {"fail_iterations": 250, "max_epochs": 100000},
    "snapshotter": {"prefix": "wine_relu"},
    "loader": {"minibatch_size": 10, "force_numpy": False},
    "learning_rate": 0.03,
    "weights_decay": 0.0,
    "layers": [10, 3]})


class WineReluWorkflow(nn_units.NNWorkflow):
    """
    Model created for class of wine recognition. Database - Wine. Model -
    fully-connected Neural Network with SoftMax loss function with RELU
    activation.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        kwargs["layers"] = layers
        super(WineReluWorkflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = WineLoader(
            self, minibatch_size=root.wine_relu.loader.minibatch_size,
            force_numpy=root.wine_relu.loader.force_numpy)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.forwards[:]
        for i, layer in enumerate(layers):
            if i < len(layers) - 1:
                aa = all2all.All2AllRELU(self, output_sample_shape=[layer])
            else:
                aa = all2all.All2AllSoftmax(
                    self, output_sample_shape=[layer])
            self.forwards.append(aa)
            if i:
                self.forwards[i].link_from(self.forwards[i - 1])
                self.forwards[i].link_attrs(
                    self.forwards[i - 1], ("input", "output"))
            else:
                self.forwards[i].link_from(self.loader)
                self.forwards[i].link_attrs(
                    self.loader, ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self)
        self.evaluator.link_from(self.forwards[-1])
        self.evaluator.link_attrs(self.forwards[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("labels", "minibatch_labels"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.DecisionGD(
            self,
            max_epochs=root.wine_relu.decision.max_epochs,
            fail_iterations=root.wine_relu.decision.fail_iterations)
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
            self, **root.wine_relu.snapshotter.__content__)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = ~self.decision.epoch_ended
        self.snapshotter.skip = ~self.decision.improved
        self.snapshotter.gate_block = self.decision.complete

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend((None,) * len(self.forwards))
        self.gds[-1] = gd.GDSoftmax(self) \
            .link_attrs(self.forwards[-1], "output", "input", "weights",
                        "bias") \
            .link_attrs(self.evaluator, "err_output") \
            .link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        self.gds[-1].gate_block = self.decision.complete
        for i in range(len(self.forwards) - 2, -1, -1):
            self.gds[i] = gd.GDRELU(self)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.forwards[i], "output", "input",
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader, ("batch_size",
                                                 "minibatch_size"))
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].gate_skip = self.decision.gd_skip

        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.repeater.gate_block = self.decision.complete

        self.gds[-1].link_from(self.decision)

    def initialize(self, learning_rate, weights_decay, device, **kwargs):
        super(WineReluWorkflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            device=device, **kwargs)


def run(load, main):
    load(WineReluWorkflow, layers=root.wine_relu.layers)
    main(learning_rate=root.wine_relu.learning_rate,
         weights_decay=root.wine_relu.weights_decay)
