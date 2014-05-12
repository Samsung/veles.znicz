#!/usr/bin/python3.3 -O
# encoding: utf-8
"""
Created on Apr 18, 2014

This workflow should clone the Imagenet example in CAFFE tool.
"""

from veles.config import root
from veles.znicz import conv, pooling, all2all, evaluator, decision
from veles.znicz import normalization, dropout
from veles.znicz.samples.imagenet.loader import LoaderDetection
from veles.znicz.standard_workflow import StandardWorkflow

import logging

root.defaults = {"all2all": {"weights_magnitude": 0.05},
                 "decision": {"fail_iterations": 100,
                              "snapshot_prefix": "imagenet_caffe",
                              "store_samples_mse": True},
                 "loader": {"minibatch_maxsize": 60},
                 "imagenet_caffe": {"learning_rate": 0.01,
                                    "weights_decay": 0.0}}


class Workflow(StandardWorkflow):
    """Workflow for ImageNet dataset.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        kwargs["name"] = kwargs.get("name", "ImageNet")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = LoaderDetection(self,
                                      ipath="/data/imagenet/2013",
                                      dbpath="/data/imagenet/2013/db",
                                      year="2013", series="img")
        self.loader.setup(level=logging.DEBUG)
        self.loader.load_data()

        self.loader.link_from(self.repeater)

        # FORWARD LAYERS

        # Layer 1 (CONV + POOL)
        conv1 = conv.ConvRELU(self, n_kernels=96, kx=11, ky=11,
                              sliding=(4, 4), padding=(0, 0, 0, 0),
                              weights_filling="gaussian", weights_stddev=0.01,
                              device=device)
        self._add_forward_unit(conv1)

        pool1 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
                                   device=device)
        self._add_forward_unit(pool1)

        norm1 = normalization.LRNormalizerForward(self, device=device)
        self._add_forward_unit(norm1)

        # Layer 2 (CONV + POOL)
        conv2 = conv.ConvRELU(self, n_kernels=256, kx=5, ky=5,
                              sliding=(1, 1), padding=(2, 2, 2, 2),
                              weights_filling="gaussian", weights_stddev=0.01,
                              device=device)
        self._add_forward_unit(conv2)

        pool2 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
                                   device=device)
        self._add_forward_unit(pool2)

        # Layer 3 (CONV)
        conv3 = conv.ConvRELU(self, n_kernels=384, kx=3, ky=3,
                              sliding=(1, 1), padding=(1, 1, 1, 1),
                              weights_filling="gaussian", weights_stddev=0.01,
                              device=device)
        self._add_forward_unit(conv3)

        # Layer 4 (CONV)
        conv4 = conv.ConvRELU(self, n_kernels=384, kx=3, ky=3,
                              sliding=(1, 1), padding=(1, 1, 1, 1),
                              weights_filling="gaussian", weights_stddev=0.01,
                              device=device)
        self._add_forward_unit(conv4)

        # Layer 5 (CONV + POOL)
        conv5 = conv.ConvRELU(self, n_kernels=256, kx=3, ky=3,
                              sliding=(1, 1), padding=(1, 1, 1, 1),
                              weights_filling="gaussian", weights_stddev=0.01,
                              device=device)
        self._add_forward_unit(conv5)

        pool5 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
                                   device=device)
        self._add_forward_unit(pool5)

        # Layer 6 (FULLY CONNECTED + 50% dropout)
        fc6 = all2all.All2AllRELU(
            self, output_shape=4096, weights_filling="gaussian",
            weights_stddev=0.005, device=device)
        self._add_forward_unit(fc6)

        drop6 = dropout.DropoutForward(self, dropout_ratio=0.5, device=device)
        self._add_forward_unit(drop6)

        # Layer 7 (FULLY CONNECTED + 50% dropout)
        fc7 = all2all.All2AllRELU(
            self, output_shape=4096, weights_filling="gaussian",
            weights_stddev=0.005, device=device)
        self._add_forward_unit(fc7)

        drop7 = dropout.DropoutForward(self, dropout_ratio=0.5, device=device)
        self._add_forward_unit(drop7)

        # LAYER 8 (FULLY CONNECTED + SOFTMAX)
        fc8sm = all2all.All2AllSoftmax(
            self, output_shape=1000, weights_filling="gaussian",
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

        # repeater and gate block
        self.repeater.link_from(self.gds[0])
        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete
        self.loader.gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device):
        super(Workflow, self).initialize(learning_rate=learning_rate,
                                         weights_decay=weights_decay,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.imagenet_caffe.layers)
    main(learning_rate=root.imagenet_caffe.learning_rate,
         weights_decay=root.imagenet_caffe.weights_decay)
