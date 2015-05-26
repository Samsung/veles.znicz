# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mar 20, 2013

Model created for digits recognition. Database - MNIST. Self-constructing
Model. It means that Model can change for any Model (Convolutional, Fully
connected, different parameters) in configuration file.

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


import os
import sys

from veles.znicz.standard_workflow import StandardWorkflow
from veles.config import root
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.conv as conv
import veles.znicz.all2all as all2all

sys.path.append(os.path.dirname(__file__))
from .loader_mnist import MnistLoader  # pylint: disable=W0611


class MnistWorkflow(StandardWorkflow):
    """Model created for digits recognition. Database - MNIST.
    Self-constructing Model. It means that Model can change for any Model
    (Convolutional, Fully connected, different parameters) in configuration
    file.
    """
    def link_mnist_weights_plotter(self, layers, limit, weights_input, parent):
        self.mnist_weights_plotter = []
        prev_channels = 1
        prev = parent
        for i in range(len(layers)):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All)):
                continue
            nme = "%s %s" % (i + 1, layers[i]["type"])
            self.debug("Added: %s", nme)
            mnist_weights_plotter = nn_plotting_units.Weights2D(
                self, name=nme, limit=limit)
            self.mnist_weights_plotter.append(mnist_weights_plotter)
            self.mnist_weights_plotter[-1].link_attrs(
                self.gds[i], ("input", weights_input))
            self.mnist_weights_plotter[-1].input_field = "mem"
            if isinstance(self.forwards[i], conv.Conv):
                self.mnist_weights_plotter[-1].get_shape_from = (
                    [self.forwards[i].kx, self.forwards[i].ky, prev_channels])
                prev_channels = self.forwards[i].n_kernels
            if (layers[i].get("output_sample_shape") is not None and
                    layers[i]["type"] != "softmax"):
                self.mnist_weights_plotter[-1].link_attrs(
                    self.forwards[i], ("get_shape_from", "input"))
            self.mnist_weights_plotter[-1].link_from(prev)
            prev = self.mnist_weights_plotter[-1]
            ee = ~self.decision.epoch_ended
            self.mnist_weights_plotter[-1].gate_skip = ee
        return prev

    def create_workflow(self):
        self.link_repeater(self.start_point)

        # Use Avatar:
        self.link_loader(self.start_point)
        self.link_avatar()
        # or just Loader:
        # self.link_loader(self.repeater)

        self.link_forwards(("input", "minibatch_data"), self.loader)
        self.link_evaluator(self.forwards[-1])
        self.link_decision(self.evaluator)
        end_units = [link(self.decision)
                     for link in (self.link_snapshotter,
                                  self.link_error_plotter,
                                  self.link_conf_matrix_plotter,
                                  self.link_err_y_plotter)]
        self.link_gds(*end_units)
        self.link_end_point(self.gds[0])
        if root.mnistr.lr_adjuster.do:
            last = self.link_lr_adjuster(self.gds[0])
        else:
            last = self.gds[0]
        self.link_end_point(last)
        self.repeater.link_from(self.link_mnist_weights_plotter(
            root.mnistr.layers, root.mnistr.weights_plotter.limit,
            "gradient_weights", last))


def run(load, main):
    load(MnistWorkflow,
         decision_config=root.mnistr.decision,
         snapshotter_config=root.mnistr.snapshotter,
         loader_name=root.mnistr.loader_name,
         loader_config=root.mnistr.loader,
         layers=root.mnistr.layers,
         loss_function=root.mnistr.loss_function,
         lr_adjuster_config=root.mnistr.lr_adjuster)
    main()
