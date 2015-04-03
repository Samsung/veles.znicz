#!/usr/bin/python3 -O
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Mar 20, 2013

Model created for digits recognition. Database - MNIST. Model - autoencoder.

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

from veles.config import root
import veles.plotting_units as plotting_units
from veles.znicz.nn_units import NNSnapshotter
import veles.znicz.nn_units as nn_units
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.conv as conv
import veles.znicz.deconv as deconv
import veles.znicz.gd_deconv as gd_deconv
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.pooling as pooling
import veles.znicz.gd_pooling as gd_pooling

sys.path.append(os.path.dirname(__file__))
from .loader_mnist import MnistLoader


root.mnist_ae.update({
    "all2all": {"weights_stddev": 0.05},
    "decision": {"fail_iterations": 20,
                 "max_epochs": 1000000000},
    "snapshotter": {"prefix": "mnist", "time_interval": 0, "compress": ""},
    "loader": {"minibatch_size": 100, "force_cpu": False,
               "normalization_type": "linear"},
    "learning_rate": 0.000001,
    "weights_decay": 0.00005,
    "gradient_moment": 0.00001,
    "weights_plotter": {"limit": 16},
    "pooling": {"kx": 3, "ky": 3, "sliding": (2, 2)},
    "include_bias": False,
    "unsafe_padding": True,
    "n_kernels": 5,
    "kx": 5,
    "ky": 5})


class MnistAELoader(MnistLoader):
    def load_data(self):
        super(MnistAELoader, self).load_data()
        self.original_data.mem.shape = [70000, 28, 28, 1]


class MnistAEWorkflow(nn_units.NNWorkflow):
    """Model created for digits recognition. Database - MNIST.
    Model - autoencoder.
    """
    def __init__(self, workflow, **kwargs):
        super(MnistAEWorkflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = MnistAELoader(
            self, **root.mnist_ae.loader.__content__)
        self.loader.link_from(self.repeater)

        unit = conv.Conv(
            self, n_kernels=root.mnist_ae.n_kernels,
            kx=root.mnist_ae.kx, ky=root.mnist_ae.ky,
            weights_filling="uniform", include_bias=root.mnist_ae.include_bias)
        unit.link_from(self.loader)
        unit.link_attrs(self.loader, ("input", "minibatch_data"))
        self.conv = unit

        unit = pooling.StochasticAbsPooling(
            self, kx=root.mnist_ae.pooling.kx, ky=root.mnist_ae.pooling.ky,
            sliding=root.mnist_ae.pooling.sliding)
        unit.link_from(self.conv)
        unit.link_attrs(self.conv, ("input", "output"))
        self.pool = unit

        unit = gd_pooling.GDMaxAbsPooling(
            self)
        unit.link_attrs(self.pool, "kx", "ky")
        unit.sliding = root.mnist_ae.pooling.sliding
        unit.link_from(self.pool)
        unit.link_attrs(self.pool, "input", "input_offset",
                        ("err_output", "output"))
        self.depool = unit

        unit = deconv.Deconv(
            self, unsafe_padding=root.mnist_ae.unsafe_padding)
        self.deconv = unit
        unit.link_from(self.depool)
        unit.link_attrs(self.conv, "weights")
        unit.link_conv_attrs(self.conv)
        unit.link_attrs(self.depool, ("input", "err_input"))
        unit.link_attrs(self.conv, ("output_shape_source", "input"))

        # Add evaluator for single minibatch
        unit = evaluator.EvaluatorMSE(self)
        self.evaluator = unit
        unit.link_from(self.deconv)
        unit.link_attrs(self.deconv, "output")
        unit.link_attrs(self.loader,
                        ("batch_size", "minibatch_size"),
                        ("target", "minibatch_data"))

        # Add decision unit
        unit = decision.DecisionMSE(
            self, fail_iterations=root.mnist_ae.decision.fail_iterations,
            max_epochs=root.mnist_ae.decision.max_epochs)
        self.decision = unit
        unit.link_from(self.evaluator)
        unit.link_attrs(self.loader, "minibatch_class",
                        "minibatch_size", "last_minibatch",
                        "class_lengths", "epoch_ended",
                        "epoch_number")
        unit.link_attrs(self.evaluator, ("minibatch_metrics", "metrics"))

        unit = NNSnapshotter(
            self, prefix=root.mnist_ae.snapshotter.prefix,
            directory=root.common.snapshot_dir,
            compress=root.mnist_ae.snapshotter.compress,
            time_interval=root.mnist_ae.snapshotter.time_interval,
            interval=root.mnist_ae.snapshotter.interval)
        self.snapshotter = unit
        unit.link_from(self.decision)
        unit.link_attrs(self.decision, ("suffix", "snapshot_suffix"))
        unit.gate_skip = ~self.loader.epoch_ended | ~self.decision.improved

        self.end_point.link_from(self.snapshotter)
        self.end_point.gate_block = ~self.decision.complete

        # Add gradient descent units
        unit = gd_deconv.GDDeconv(
            self,
            learning_rate=root.mnist_ae.learning_rate,
            weights_decay=root.mnist_ae.weights_decay,
            gradient_moment=root.mnist_ae.gradient_moment)
        self.gd_deconv = unit
        unit.link_attrs(self.evaluator, "err_output")
        unit.link_attrs(
            self.deconv, "weights", "input", "hits", "n_kernels",
            "kx", "ky", "sliding", "padding", "unpack_data", "unpack_size")
        unit.gate_skip = self.decision.gd_skip
        unit.need_err_input = False
        unit.gate_block = self.decision.complete
        self.repeater.link_from(unit)

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        prev = self.snapshotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(prev)
            self.plt[-1].gate_skip = ~self.decision.epoch_ended
            self.plt[-1].gate_block = self.decision.complete
            prev = self.plt[-1]
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True

        # Weights plotter
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="Weights", limit=root.mnist_ae.weights_plotter.limit)
        self.plt_mx.link_attrs(self.conv, ("input", "weights"))
        self.plt_mx.get_shape_from = [root.mnist_ae.kx,
                                      root.mnist_ae.ky, 1]
        self.plt_mx.input_field = "mem"
        self.plt_mx.link_from(prev)
        self.plt_mx.gate_skip = ~self.decision.epoch_ended
        self.plt_mx.gate_block = self.decision.complete
        prev = self.plt_mx

        # Input plotter

        self.plt_inp = nn_plotting_units.Weights2D(
            self, name="First Layer Input",
            limit=root.mnist_ae.weights_plotter.limit)
        self.plt_inp.link_attrs(self.conv, "input")
        self.plt_inp.get_shape_from = [28, 28, 1]
        self.plt_inp.input_field = "mem"
        self.plt_inp.link_from(prev)
        self.plt_inp.gate_skip = ~self.decision.epoch_ended
        self.plt_inp.gate_block = self.decision.complete
        prev = self.plt_inp

        # Output plotter
        self.plt_out = nn_plotting_units.Weights2D(
            self, name="First Layer Output",
            limit=root.mnist_ae.weights_plotter.limit)
        self.plt_out.link_attrs(self.conv, ("input", "output"))
        self.plt_out.get_shape_from = [
            28 - root.mnist_ae.kx + 1,
            28 - root.mnist_ae.ky + 1,
            root.mnist_ae.n_kernels]
        self.plt_out.input_field = "mem"
        self.plt_out.link_from(prev)
        self.plt_out.gate_skip = ~self.decision.epoch_ended
        self.plt_out.gate_block = self.decision.complete
        prev = self.plt_out

        # Deconv result plotter
        self.plt_out = nn_plotting_units.Weights2D(
            self, name="Deconv result",
            limit=root.mnist_ae.weights_plotter.limit)
        self.plt_out.link_attrs(self.deconv, ("input", "output"))
        self.plt_out.get_shape_from = [28, 28, 1]
        self.plt_out.input_field = "mem"
        self.plt_out.link_from(prev)
        self.plt_out.gate_skip = ~self.decision.epoch_ended
        self.plt_out.gate_block = self.decision.complete
        prev = self.plt_out

        self.gd_deconv.link_from(prev)


def run(load, main):
    load(MnistAEWorkflow)
    main()
