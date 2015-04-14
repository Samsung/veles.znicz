# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on August 12, 2013

Model created for digits recognition. Database - MNIST. Model - fully-connected
Neural Network with MSE loss function with target encoded as 7 points.

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

import numpy
from zope.interface import implementer

from veles.config import root
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.loader as loader
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.nn_units import NNSnapshotter


sys.path.append(os.path.dirname(__file__))
from .loader_mnist import MnistLoader


root.mnist7.update({
    "decision": {"fail_iterations": 25, "max_epochs": 1000000},
    "snapshotter": {"prefix": "mnist7"},
    "loader": {"minibatch_size": 60, "force_numpy": False},
    "weights_plotter": {"limit": 25},
    "learning_rate": 0.0000016,
    "weights_decay": 0.00005,
    "layers": [100, 100, 7]})


@implementer(loader.IFullBatchLoader)
class Mnist7Loader(MnistLoader, loader.FullBatchLoaderMSE):
    """Loads MNIST dataset.
    """
    def load_data(self):
        """Here we will load MNIST data.
        """
        super(Mnist7Loader, self).load_data()
        self.class_targets.reset()
        self.class_targets.mem = numpy.array(
            [[1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],  # 0
             [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0],  # 1
             [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0],  # 2
             [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0],  # 3
             [-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0],  # 4
             [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],  # 5
             [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],  # 6
             [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0],  # 7
             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 8
             [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0]],  # 9)
            dtype=opencl_types.dtypes[root.common.precision_type])
        self.original_targets.mem = numpy.zeros(
            (len(self.original_labels), 7),
            dtype=self.original_data.dtype)
        for i, label in enumerate(self.original_labels):
            self.original_targets[i] = self.class_targets[label]


class Mnist7Workflow(nn_units.NNWorkflow):
    """
    Model created for digits recognition. Database - MNIST.
    Model - fully-connected Neural Network with MSE loss function with target
    encoded as 7 points.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        kwargs["layers"] = layers
        super(Mnist7Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Mnist7Loader(
            self, minibatch_size=root.mnist7.loader.minibatch_size,
            normalization_type=root.mnist7.loader.normalization_type,
            force_numpy=root.mnist7.loader.force_numpy)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.forwards[:]
        for i, layer in enumerate(layers):
            aa = all2all.All2AllTanh(self, output_sample_shape=[layer])
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
        self.evaluator = evaluator.EvaluatorMSE(self)
        self.evaluator.link_from(self.forwards[-1])
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("target", "minibatch_targets"),
                                  ("labels", "minibatch_labels"),
                                  "class_targets")
        self.evaluator.link_attrs(self.forwards[-1], "output")

        # Add decision unit
        self.decision = decision.DecisionMSE(
            self, fail_iterations=root.mnist7.decision.fail_iterations,
            max_epochs=root.mnist7.decision.max_epochs)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class", "minibatch_size",
                                 "last_minibatch", "class_lengths",
                                 "epoch_ended", "epoch_number",
                                 "minibatch_offset", "minibatch_size")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_metrics", "metrics"),
            ("minibatch_mse", "mse"))

        self.snapshotter = NNSnapshotter(
            self, prefix=root.mnist7.snapshotter.prefix,
            directory=root.common.snapshot_dir,
            interval=root.mnist7.snapshotter.interval,
            time_interval=root.mnist7.snapshotter.time_interval)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = ~self.loader.epoch_ended
        self.snapshotter.skip = ~self.decision.improved
        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self)
        self.image_saver.link_from(self.snapshotter)
        self.image_saver.link_attrs(self.evaluator, "output", "target")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indices", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")
        self.image_saver.gate_skip = ~self.decision.improved
        self.image_saver.link_attrs(self.snapshotter,
                                    ("this_save_time", "time"))

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(None for i in range(0, len(self.forwards)))
        self.gds[-1] = gd.GDTanh(self)
        self.gds[-1].link_from(self.image_saver)
        self.gds[-1].link_attrs(self.forwards[-1], "output", "input",
                                "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.forwards) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.forwards[i], "output", "input",
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader, ("batch_size",
                                                 "minibatch_size"))
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.repeater.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else
                                   self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True
        self.plt[0].gate_block = self.decision.complete
        # Weights plotter
        # """
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="First Layer Weights",
            limit=root.mnist7.weights_plotter.limit)
        self.plt_mx.link_attrs(self.gds[0], ("input", "weights"))
        self.plt_mx.input_field = "mem"
        self.plt_mx.link_attrs(self.forwards[0], ("get_shape_from", "input"))
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = \
            ~self.decision.epoch_ended | self.decision.complete
        # """
        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(3):
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_max[-1].link_attrs(self.decision,
                                        ("input", "epoch_n_err_pt"))
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offset = 1
            self.plt_max[-1].link_from(self.plt[-1] if not i else
                                       self.plt_max[-2])
        self.plt_max[0].gate_block = self.decision.complete
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(3):
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].link_attrs(self.decision,
                                        ("input", "epoch_n_err_pt"))
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offset = 2
            self.plt_min[-1].link_from(self.plt_max[-1] if not i else
                                       self.plt_min[-2])
        self.plt_min[-1].redraw_plot = True
        self.plt_min[0].gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device, snapshot=False,
                   **kwargs):
        super(Mnist7Workflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            snapshot=False, device=device)


def run(load, main):
    load(Mnist7Workflow, layers=root.mnist7.layers)
    main(learning_rate=root.mnist7.learning_rate,
         weights_decay=root.mnist7.weights_decay)
