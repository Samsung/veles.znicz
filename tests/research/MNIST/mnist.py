#!/usr/bin/python3 -O
"""
Created on Mar 20, 2013

Model created for digits recognition. Database - MNIST. Self-constructing
Model. It means that Model can change for any Model (Convolutional, Fully
connected, different parameters) in configuration file.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.znicz.standard_workflow import StandardWorkflow
from veles.config import root
from veles.genetics import Tune
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.conv as conv
import veles.znicz.all2all as all2all
import veles.znicz.lr_adjust as lra
from .loader_mnist import MnistLoader

root.mnistr.update({
    "learning_rate_adjust": {"do": False},
    "add_plotters": True,
    "loss_function": "softmax",
    "decision": {"fail_iterations": 100,
                 "max_epochs": 1000000000},
    "snapshotter": {"prefix": "mnist", "time_interval": 0, "compress": "",
                    "interval": 1},
    "loader": {"minibatch_size": Tune(60, 1, 1000), "on_device": True},
    "weights_plotter": {"limit": 64},
    "layers": [{"type": "all2all_tanh", "output_shape": Tune(100, 10, 500),
                "learning_rate": Tune(0.03, 0.0001, 0.9),
                "weights_decay": Tune(0.0, 0.0, 0.9),
                "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                "weights_decay_bias": Tune(0.0, 0.0, 0.9),
                "gradient_moment": Tune(0.0, 0.0, 0.95),
                "gradient_moment_bias": Tune(0.0, 0.0, 0.95),
                "factor_ortho": Tune(0.001, 0.0, 0.1),
                "weights_filling": "uniform",
                "weights_stddev": Tune(0.05, 0.0001, 0.1),
                "bias_filling": "uniform",
                "bias_stddev": Tune(0.05, 0.0001, 0.1)},
               {"type": "softmax", "output_shape": 10,
                "learning_rate": Tune(0.03, 0.0001, 0.9),
                "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                "weights_decay": Tune(0.0, 0.0, 0.95),
                "weights_decay_bias": Tune(0.0, 0.0, 0.95),
                "gradient_moment": Tune(0.0, 0.0, 0.95),
                "gradient_moment_bias": Tune(0.0, 0.0, 0.95),
                "weights_filling": "uniform",
                "weights_stddev": Tune(0.05, 0.0001, 0.1),
                "bias_filling": "uniform",
                "bias_stddev": Tune(0.05, 0.0001, 0.1)}]})


class MnistWorkflow(StandardWorkflow):
    """Model created for digits recognition. Database - MNIST.
    Self-constructing Model. It means that Model can change for any Model
    (Convolutional, Fully connected, different parameters) in configuration
    file.
    """
    def link_mnist_lr_adjuster(self, init_unit):
        self.mnist_lr_adjuster = lra.LearningRateAdjust(self)
        for gd_elm in self.gds:
            self.mnist_lr_adjuster.add_gd_unit(
                gd_elm,
                lr_policy=lra.InvAdjustPolicy(0.01, 0.0001, 0.75),
                bias_lr_policy=lra.InvAdjustPolicy(0.01, 0.0001, 0.75))
        self.mnist_lr_adjuster.link_from(init_unit)

    def link_mnist_weights_plotter(self, init_unit, layers, limit,
                                   weights_input):
        self.mnist_weights_plotter = []
        prev_channels = 1
        prev = init_unit
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
            if (layers[i].get("output_shape") is not None and
                    layers[i]["type"] != "softmax"):
                self.mnist_weights_plotter[-1].link_attrs(
                    self.forwards[i], ("get_shape_from", "input"))
            self.mnist_weights_plotter[-1].link_from(prev)
            prev = self.mnist_weights_plotter[-1]
            ee = ~self.decision.epoch_ended
            self.mnist_weights_plotter[-1].gate_skip = ee

    def link_loader(self, init_unit):
        self.loader = MnistLoader(
            self,
            minibatch_size=root.mnistr.loader.minibatch_size,
            on_device=root.mnistr.loader.on_device)
        self.loader.link_from(init_unit)

    def create_workflow(self):
        # Add repeater unit
        self.link_repeater(self.start_point)

        # Add loader unit
        self.link_loader(self.repeater)

        # Add fwds units
        self.link_forwards(self.loader, ("input", "minibatch_data"))

        # Add evaluator unit
        self.link_evaluator(self.forwards[-1])

        # Add decision unit
        self.link_decision(self.evaluator)

        # Add snapshotter unit
        self.link_snapshotter(self.decision)

        # Add gradient descent units
        self.link_gds(self.snapshotter)

        if root.mnistr.add_plotters:

            # Add error plotter unit
            self.link_error_plotter(self.gds[0])

            # Add Confusion matrix plotter unit
            self.link_conf_matrix_plotter(self.error_plotter[-1])

            # Add Err y plotter unit
            self.link_err_y_plotter(self.conf_matrix_plotter[-1])

            # Add Weights plotter unit
            self.link_mnist_weights_plotter(
                self.err_y_plotter[-1], layers=root.mnistr.layers,
                limit=root.mnistr.weights_plotter.limit,
                weights_input="gradient_weights")

            last = self.mnist_weights_plotter[-1]
        else:
            last = self.gds[0]

        if root.mnistr.learning_rate_adjust.do:

            # Add learning_rate_adjust unit
            self.link_mnist_lr_adjuster(last)

            # Add end_point unit
            self.link_end_point(self.mnist_lr_adjuster)
        else:
            # Add end_point unit
            self.link_end_point(last)


def run(load, main):
    load(MnistWorkflow, layers=root.mnistr.layers,
         fail_iterations=root.mnistr.decision.fail_iterations,
         max_epochs=root.mnistr.decision.max_epochs,
         prefix=root.mnistr.snapshotter.prefix,
         snapshot_interval=root.mnistr.snapshotter.interval,
         snapshot_dir=root.common.snapshot_dir,
         loss_function=root.mnistr.loss_function)
    main()
