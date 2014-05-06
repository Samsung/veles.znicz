#!/usr/bin/python3.3 -O
# encoding: utf-8
"""
Created on May 6, 2014

A workflow to test first layer in simple line detection.
"""


import logging
import sys

from veles.config import root
from veles.znicz import conv, pooling, all2all, evaluator, decision
from veles.znicz.standard_workflow import StandardWorkflow
from veles.znicz.loader import ImageLoader
from veles.znicz.nn_plotting_units import Weights2D
from enum import IntEnum


root.defaults = {"all2all": {"weights_magnitude": 0.05},
                 "decision": {"fail_iterations": 100,
                              "snapshot_prefix": "lines",
                              "store_samples_mse": True},
                 "loader": {"minibatch_maxsize": 60},
                 "weights_plotter": {"limit": 32},
                 "lines": {"global_alpha": 0.01,
                           "global_lambda": 0.0}}


class ImageLabel(IntEnum):
    """Enum for different types of lines"""
    vertical = 0
    horizontal = 1
    tilted_bottom_to_top = 2  # left lower --> right top
    tilted_top_to_bottom = 3  # left top --> right bottom


class Loader(ImageLoader):
    def get_label_from_filename(self, filename):
        #takes folder name "vertical", "horizontal", "etc"
        return int(ImageLabel[filename.split("/")[-2]])


class Workflow(StandardWorkflow):
    """Workflow for Lines dataset.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        kwargs["name"] = kwargs.get("name", "Lines")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)
        self.loader = Loader(
            self, train_paths=["/data/veles/Lines/LINES_10_500_NOISY/learning"],
            validation_paths=["/data/veles/Lines/LINES_10_500_NOISY/test"],
            minibatch_maxsize=root.loader.minibatch_maxsize,
            grayscale=False)

        self.loader.setup(level=logging.DEBUG)
        self.loader.load_data()

        self.loader.link_from(self.repeater)

        # FORWARD LAYERS

        # Layer 1 (CONV + POOL)
        conv1 = conv.ConvRELU(self, n_kernels=32, kx=11, ky=11,
                              sliding=(4, 4), padding=(0, 0, 0, 0),
                              weights_filling="gaussian", weights_stddev=1.0,
                              device=device)
        self._add_forward_unit(conv1)

        pool1 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
                                   device=device)
        self._add_forward_unit(pool1)

        # Layer 7 (FULLY CONNECTED)
#        fc7 = all2all.All2AllRELU(
#            self, output_shape=100, weights_filling="gaussian",
#            weights_stddev=0.005, device=device)
#        self._add_forward_unit(fc7)

        # LAYER 8 (FULLY CONNECTED + SOFTMAX)
        fc8sm = all2all.All2AllSoftmax(
            self, output_shape=len(ImageLabel), weights_filling="gaussian",
            weights_stddev=0.01, device=device)
        self._add_forward_unit(fc8sm)

        # EVALUATOR
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(fc8sm)
        self.evaluator.link_attrs(fc8sm, ("y", "output"), "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("labels", "minibatch_labels"))

        # Add decision unit
        self.decision = decision.Decision(self)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "class_samples",
                                 "no_more_minibatches_left")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_y_sum"))

        # BACKWARD LAYERS (GRADIENT DESCENT)
        self._create_gradient_descent_units()

        # Weights plotter
        self.decision.vectors_to_sync[self.gds[0].weights] = 1
        self.plt_mx = Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit)
        self.plt_mx.link_attrs(self.gds[0], ("input", "weights"))
        self.plt_mx.input_field = "v"
        self.plt_mx.get_shape_from = (
            [self.fwds[0].kx, self.fwds[0].ky, 3]
            if isinstance(self.fwds[0], conv.Conv)
            else self.fwds[0].input)
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended

        # repeater and gate block
        self.repeater.link_from(self.gds[0])
        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete
        self.loader.gate_block = self.decision.complete

    def initialize(self, global_alpha, global_lambda, device):
        super(Workflow, self).initialize(global_alpha=global_alpha,
                                         global_lambda=global_lambda,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.lines.layers)
    main(global_alpha=root.lines.global_alpha,
         global_lambda=root.lines.global_lambda)
