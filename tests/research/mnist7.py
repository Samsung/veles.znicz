#!/usr/bin/python3 -O
"""
Created on August 12, 2013

Model created for digits recognition. Database – MNIST. Model – fully-connected
Neural Network with MSE loss function with target encoded as 7 points.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
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
import veles.znicz.loader as loader
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.nn_units import NNSnapshotter
import veles.znicz.samples.mnist as mnist


mnist_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "samples/MNIST")
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")

root.mnist7.update({
    "decision": {"fail_iterations": 25, "max_epochs": 1000000},
    "snapshotter": {"prefix": "mnist7"},
    "loader": {"minibatch_size": 60, "on_device": True},
    "weights_plotter": {"limit": 25},
    "learning_rate": 0.0000016,
    "weights_decay": 0.00005,
    "layers": [100, 100, 7],
    "data_paths": {"test_images": test_image_dir,
                   "test_label": test_label_dir,
                   "train_images": train_image_dir,
                   "train_label": train_label_dir}})


@implementer(loader.IFullBatchLoader)
class Mnist7Loader(mnist.MnistLoader, loader.FullBatchLoaderMSE):
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
            [self.original_labels.shape[0], 7],
            dtype=self.original_data.dtype)
        for i in range(0, self.original_labels.shape[0]):
            label = self.original_labels[i]
            self.original_targets[i] = self.class_targets[label]


class Mnist7Workflow(nn_units.NNWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Mnist7Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Mnist7Loader(
            self, minibatch_size=root.mnist7.loader.minibatch_size,
            on_device=root.mnist7.loader.on_device)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.fwds[:]
        for i in range(0, len(layers)):
            aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                     device=device)
            self.fwds.append(aa)
            if i:
                self.fwds[i].link_from(self.fwds[i - 1])
                self.fwds[i].link_attrs(self.fwds[i - 1],
                                        ("input", "output"))
            else:
                self.fwds[i].link_from(self.loader)
                self.fwds[i].link_attrs(self.loader,
                                        ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorMSE(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("target", "minibatch_targets"),
                                  ("labels", "minibatch_labels"),
                                  "class_targets")
        self.evaluator.link_attrs(self.fwds[-1], "output")

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
            directory=root.common.snapshot_dir)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.decision.epoch_ended | ~self.decision.improved)
        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self)
        self.image_saver.link_from(self.snapshotter)
        self.image_saver.link_attrs(self.evaluator, "output", "target")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indexes", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")
        self.image_saver.gate_skip = ~self.decision.improved
        self.image_saver.link_attrs(self.snapshotter,
                                    ("this_save_time", "time"))

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(None for i in range(0, len(self.fwds)))
        self.gds[-1] = gd.GDTanh(self, device=device)
        self.gds[-1].link_from(self.image_saver)
        self.gds[-1].link_attrs(self.fwds[-1], "output", "input",
                                "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.fwds[i], "output", "input",
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader, ("batch_size",
                                                 "minibatch_size"))
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else
                                   self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True
        # Weights plotter
        # """
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="First Layer Weights",
            limit=root.mnist7.weights_plotter.limit)
        self.plt_mx.link_attrs(self.gds[0], ("input", "weights"))
        self.plt_mx.input_field = "mem"
        self.plt_mx.link_attrs(self.fwds[0], ("get_shape_from", "input"))
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended
        # """
        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(0, 3):
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_max[-1].link_attrs(self.decision,
                                        ("input", "epoch_n_err_pt"))
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offset = 1
            self.plt_max[-1].link_from(self.plt[-1] if not i else
                                       self.plt_max[-2])
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(0, 3):
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].link_attrs(self.decision,
                                        ("input", "epoch_n_err_pt"))
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offset = 2
            self.plt_min[-1].link_from(self.plt_max[-1] if not i else
                                       self.plt_min[-2])
        self.plt_min[-1].redraw_plot = True

    def initialize(self, learning_rate, weights_decay, device, **kwargs):
        super(Mnist7Workflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            device=device)


def run(load, main):
    load(Mnist7Workflow, layers=root.mnist7.layers)
    main(learning_rate=root.mnist7.learning_rate,
         weights_decay=root.mnist7.weights_decay)
