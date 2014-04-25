#!/usr/bin/python3.3 -O
"""
Created on August 12, 2013

MNIST with target encoded as 7 points, MSE.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy

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
import veles.znicz.samples.mnist as mnist

root.defaults = {"decision": {"fail_iterations": 25,
                              "snapshot_prefix": "mnist7"},
                 "loader": {"minibatch_maxsize": 60},
                 "weights_plotter": {"limit": 25},
                 "mnist7": {"global_alpha": 0.0001,
                            "global_lambda": 0.00005,
                            "layers": [100, 100, 7]}}


class Loader(mnist.Loader):
    """Loads MNIST dataset.
    """
    def load_data(self):
        """Here we will load MNIST data.
        """
        super(Loader, self).load_data()
        self.class_target.reset()
        self.class_target.v = numpy.array(
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
            dtype=opencl_types.dtypes[root.common.dtype])
        self.original_target = numpy.zeros(
            [self.original_labels.shape[0], 7],
            dtype=opencl_types.dtypes[root.common.dtype])
        for i in range(0, self.original_labels.shape[0]):
            label = self.original_labels[i]
            self.original_target[i] = self.class_target[label]


class Workflow(nn_units.NNWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self,
                             minibatch_maxsize=root.loader.minibatch_maxsize)
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
        self.evaluator.link_attrs(self.fwds[-1], ("y", "output"))
        self.evaluator.link_attrs(self.loader,
                           ("batch_size", "minibatch_size"),
                           ("target", "minibatch_target"),
                           ("labels", "minibatch_labels"),
                           ("max_samples_per_epoch", "total_samples"),
                           "class_target")

        # Add decision unit
        self.decision = decision.Decision(
            self,
            snapshot_prefix=root.decision.snapshot_prefix,
            fail_iterations=root.decision.fail_iterations)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "no_more_minibatches_left",
                                 "class_samples")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_metrics", "metrics"))

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self)
        self.image_saver.link_from(self.decision)
        self.image_saver.link_attrs(self.evaluator,
                                    ("output", "y"), "target")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indexes", "minibatch_indexes"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")
        self.image_saver.link_attrs(self.decision,
                                    ("this_save_time", "snapshot_time"))
        self.image_saver.gate_skip = ~self.decision.just_snapshotted

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(None for i in range(0, len(self.fwds)))
        self.gds[-1] = gd.GDTanh(self, device=device)
        self.gds[-1].link_from(self.image_saver)
        self.gds[-1].link_attrs(self.fwds[-1],
                               ("y", "output"),
                               ("h", "input"),
                               "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_y")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.fwds[i],
                                  ("y", "output"),
                                  ("h", "input"),
                                  "weights", "bias")
            self.gds[i].link_attrs(self.loader, ("batch_size",
                                                "minibatch_size"))
            self.gds[i].link_attrs(self.gds[i + 1], ("err_y", "err_h"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else
                                   self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True
        # Weights plotter
        # """
        self.decision.vectors_to_sync[self.gds[0].weights] = 1
        self.plt_mx = plotting_units.Weights2D(
            self, name="First Layer Weights",
            limit=root.weights_plotter.limit)
        self.plt_mx.link_attrs(self.gds[0], ("input", "weights"))
        self.plt_mx.input_field = "v"
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
                                        ("input", "epoch_metrics"))
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.plt[-1] if not i else
                                       self.plt_max[-2])
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(0, 3):
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.plt_max[-1] if not i else
                                       self.plt_min[-2])
        self.plt_min[-1].redraw_plot = True

    def initialize(self, global_alpha, global_lambda, device, **kwargs):
        super(Workflow, self).initialize(global_alpha=global_alpha,
                                         global_lambda=global_lambda,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.mnist7.layers)
    main(global_alpha=root.mnist7.global_alpha,
         global_lambda=root.mnist7.global_lambda)
