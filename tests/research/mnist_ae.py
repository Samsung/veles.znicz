#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

File for MNIST dataset.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

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
from veles.znicz.samples.mnist import Loader


mnist_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "samples/MNIST")
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")


root.defaults = {"all2all": {"weights_stddev": 0.05},
                 "decision": {"fail_iterations": 20,
                              "store_samples_mse": True},
                 "snapshotter": {"prefix": "mnist"},
                 "loader": {"minibatch_size": 100},
                 "mnist": {"learning_rate": 0.03,
                           "weights_decay": 0.0,
                           "layers": [100, 10],
                           "data_paths": {"test_images": test_image_dir,
                                          "test_label": test_label_dir,
                                          "train_images": train_image_dir,
                                          "train_label": train_label_dir}}}


class Workflow(nn_units.NNWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    """
    def __init__(self, workflow, layers, **kwargs):
        kwargs["name"] = kwargs.get("name", "MNIST")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, name="MNIST loader",
                             minibatch_size=root.loader.minibatch_size,
                             on_device=True)
        self.loader.link_from(self.repeater)

        LR = 0.000001
        WD = 0.00005
        GM = 0.00001
        KX = 5
        KY = 5
        N_KERNELS = 5

        unit = conv.Conv(self, n_kernels=N_KERNELS, kx=KX, ky=KY,
                         weights_filling="uniform", include_bias=False)
        unit.link_from(self.loader)
        unit.link_attrs(self.loader, ("input", "minibatch_data"))
        self.conv = unit

        unit = pooling.StochasticAbsPooling(self, kx=3, ky=3, sliding=(2, 2))
        unit.link_from(self.conv)
        unit.link_attrs(self.conv, ("input", "output"))
        self.pool = unit

        unit = gd_pooling.GDMaxAbsPooling(
            self, kx=self.pool.kx, ky=self.pool.ky, sliding=(2, 2))
        unit.link_from(self.pool)
        unit.link_attrs(self.pool, "input", "input_offset",
                        ("err_output", "output"))
        self.depool = unit

        unit = deconv.Deconv(
            self, n_kernels=N_KERNELS, kx=KX, ky=KY,
            sliding=self.conv.sliding, padding=self.conv.padding)
        self.deconv = unit
        unit.link_from(self.depool)
        unit.link_attrs(self.conv, "weights")
        unit.link_attrs(self.depool, ("input", "err_input"))

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
            self, fail_iterations=root.decision.fail_iterations)
        self.decision = unit
        unit.link_from(self.evaluator)
        unit.link_attrs(self.loader, "minibatch_class",
                        "minibatch_size", "last_minibatch",
                        "class_lengths", "epoch_ended",
                        "epoch_number")
        unit.link_attrs(self.evaluator, ("minibatch_metrics", "metrics"))

        unit = NNSnapshotter(self, prefix=root.snapshotter.prefix,
                             directory=root.common.snapshot_dir,
                             compress="", time_interval=0)
        self.snapshotter = unit
        unit.link_from(self.decision)
        unit.link_attrs(self.decision, ("suffix", "snapshot_suffix"))
        unit.gate_skip = ~self.loader.epoch_ended | ~self.decision.improved

        self.end_point.link_from(self.snapshotter)
        self.end_point.gate_block = ~self.decision.complete

        # Add gradient descent units
        unit = gd_deconv.GDDeconv(
            self, n_kernels=N_KERNELS, kx=KX, ky=KY,
            sliding=self.conv.sliding, padding=self.conv.padding,
            learning_rate=LR, weights_decay=WD, gradient_moment=GM)
        self.gd_deconv = unit
        unit.link_attrs(self.evaluator, "err_output")
        unit.link_attrs(self.deconv, "weights", "input")
        unit.gate_skip = self.decision.gd_skip

        self.gd_deconv.need_err_input = False
        self.repeater.link_from(self.gd_deconv)

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
            prev = self.plt[-1]
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True

        # Weights plotter
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="Weights", limit=64)
        self.plt_mx.link_attrs(self.conv, ("input", "weights"))
        self.plt_mx.get_shape_from = [KX, KY, 1]
        self.plt_mx.input_field = "mem"
        self.plt_mx.link_from(prev)
        self.plt_mx.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_mx

        # Input plotter
        """
        self.plt_inp = nn_plotting_units.Weights2D(
            self, name="First Layer Input", limit=64)
        self.plt_inp.link_attrs(self.conv, "input")
        self.plt_inp.get_shape_from = [28, 28, 1]
        self.plt_inp.input_field = "mem"
        self.plt_inp.link_from(prev)
        self.plt_inp.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_inp
        """

        # Output plotter
        """
        self.plt_out = nn_plotting_units.Weights2D(
            self, name="First Layer Output", limit=64)
        self.plt_out.link_attrs(self.conv, ("input", "output"))
        self.plt_out.get_shape_from = [28 - KX + 1, 28 - KY + 1, N_KERNELS]
        self.plt_out.input_field = "mem"
        self.plt_out.link_from(prev)
        self.plt_out.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_out
        """

        # Deconv result plotter
        self.plt_out = nn_plotting_units.Weights2D(
            self, name="Deconv result", limit=64)
        self.plt_out.link_attrs(self.deconv, ("input", "output"))
        self.plt_out.get_shape_from = [28, 28, 1]
        self.plt_out.input_field = "mem"
        self.plt_out.link_from(prev)
        self.plt_out.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_out

        self.gd_deconv.link_from(prev)
        self.gd_deconv.gate_block = self.decision.complete


def run(load, main):
    load(Workflow, layers=root.mnist.layers)
    main()
