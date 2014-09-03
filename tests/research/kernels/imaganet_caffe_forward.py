#!/usr/bin/python3 -O
# encoding: utf-8
"""
Created on Aug 13, 2014

Data archiving.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""
from veles.config import root
from veles.znicz import conv, pooling, all2all
from veles.znicz import normalization, activation
from veles.znicz.standard_workflow import StandardWorkflow
import os
from veles.znicz.tests.research.kernels.imagenet_loader\
    import ImgLoaderClassifier


root.common.update = {"precision_type": "float", }

root.defaults = {
    "decision": {"fail_iterations": 100,
                 "store_samples_mse": True},
    "snapshotter": {"prefix": "imagenet_caffe"},
    "loader": {"minibatch_size": 256},
    "imagenet_caffe": {"learning_rate": 0.00016,
                       "weights_decay": 0.0,
                       "layers":
                       [{"type": "conv_relu", "n_kernels": 96,
                         "kx": 11, "ky": 11, "padding": (0, 0, 0, 0),
                         "sliding": (4, 4),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},
                        {"type": "max_pooling",
                         "kx": 3, "ky": 3, "sliding": (2, 2)},
                        {"type": "norm", "alpha": 0.00005,
                         "beta": 0.75, "n": 3},

                        {"type": "conv_relu", "n_kernels": 256,
                         "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                         "sliding": (1, 1),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},
                        {"type": "max_pooling",
                         "kx": 3, "ky": 3, "sliding": (2, 2)},

                        {"type": "conv", "n_kernels": 384,
                         "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                         "sliding": (1, 1),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},

                        {"type": "conv", "n_kernels": 384,
                         "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                         "sliding": (1, 1),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},

                        {"type": "conv_relu", "n_kernels": 256,
                         "kx": 3, "ky": 3, "padding": (1, 1, 1, 1),
                         "sliding": (1, 1),
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01},
                        {"type": "max_pooling",
                         "kx": 3, "ky": 3, "sliding": (2, 2)},

                        {"type": "all2all_relu", "output_shape": 4096,
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.005},

                        {"type": "dropout", "dropout_ratio": 0.5},

                        {"type": "softmax", "output_shape": 1000,
                         "weights_filling": "gaussian",
                         "weights_stddev": 0.01}]}}


class Workflow(StandardWorkflow):
    """Workflow for ImageNet dataset.
    """
    def __init__(self, workflow, **kwargs):
        n_classes = 1000
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        data_path = os.path.dirname(os.path.realpath(__file__))
        kwargs["name"] = kwargs.get("name", "ImageNet")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.loader = ImgLoaderClassifier(
            self, path_mean_img=os.path.join(data_path,
                                             "data/ilsvrc_2012_mean.npy"),
            max_minibatch_size=3, minibatch_size=3,
            test_paths=[os.path.join(data_path, "data/test")],
            validation_paths=[], train_paths=[])
        self.loader.load_data()

        self.loader.link_from(self.start_point)
        # FORWARD LAYERS

        # Layer 1 (CONV + POOL)
        conv1 = conv.Conv(
            self, name="conv1",
            n_kernels=96, kx=11, ky=11, sliding=(4, 4), padding=(0, 0, 0, 0),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=0)
        self._add_forward_unit(conv1)

        relu1 = activation.ForwardStrictRELU(self, name="relu1")
        self._add_forward_unit(relu1)

        pool1 = pooling.MaxPooling(self, name="pool1",
                                   kx=3, ky=3, sliding=(2, 2))
        self._add_forward_unit(pool1)

        norm1 = normalization.LRNormalizerForward(
            self, name="norm1", alpha=0.00002, beta=0.75, n=5, k=1)
        self._add_forward_unit(norm1)

        # Layer 2 (CONV + POOL)
        conv2 = conv.Conv(
            self, name="conv2",
            n_kernels=256, kx=5, ky=5, sliding=(1, 1), padding=(2, 2, 2, 2),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=1.0)
        self._add_forward_unit(conv2)

        relu2 = activation.ForwardStrictRELU(self, name="relu2")
        self._add_forward_unit(relu2)

        pool2 = pooling.MaxPooling(self, name="pool2",
                                   kx=3, ky=3, sliding=(2, 2))
        self._add_forward_unit(pool2)

        norm2 = normalization.LRNormalizerForward(
            self, name="norm2", alpha=0.00002, beta=0.75, n=5, k=1,)
        self._add_forward_unit(norm2)

        # Layer 3 (CONV)
        conv3 = conv.Conv(
            self, name="conv3",
            n_kernels=384, kx=3, ky=3, sliding=(1, 1), padding=(1, 1, 1, 1),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=0)
        self._add_forward_unit(conv3)

        relu3 = activation.ForwardStrictRELU(self, name="relu3")
        self._add_forward_unit(relu3)

        # Layer 4 (CONV)
        conv4 = conv.Conv(
            self, name="conv4",
            n_kernels=384, kx=3, ky=3, sliding=(1, 1), padding=(1, 1, 1, 1),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=1)
        self._add_forward_unit(conv4)

        relu4 = activation.ForwardStrictRELU(self, name="relu4")
        self._add_forward_unit(relu4)

        # Layer 5 (CONV + POOL)
        conv5 = conv.Conv(
            self, name="conv5",
            n_kernels=256, kx=3, ky=3, sliding=(1, 1), padding=(1, 1, 1, 1),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=1)
        self._add_forward_unit(conv5)

        relu5 = activation.ForwardStrictRELU(self, name="relu5")
        self._add_forward_unit(relu5)

        pool5 = pooling.MaxPooling(self, name="pool5",
                                   kx=3, ky=3, sliding=(2, 2))
        self._add_forward_unit(pool5)

        # Layer 6 (FULLY CONNECTED + 50% dropout)
        fc6 = all2all.All2All(
            self, name="fc6",
            output_shape=4096,
            weights_filling="gaussian", weights_stddev=0.005,
            bias_filling="constant", bias_stddev=1)
        self._add_forward_unit(fc6)

        relu6 = activation.ForwardStrictRELU(self, name="relu6")
        self._add_forward_unit(relu6)

        # Layer 7 (FULLY CONNECTED + 50% dropout)
        fc7 = all2all.All2All(
            self, name="fc7", output_shape=4096, weights_filling="gaussian",
            weights_stddev=0.005)
        self._add_forward_unit(fc7)

        relu7 = activation.ForwardStrictRELU(self, name="relu7")
        self._add_forward_unit(relu7)

        # LAYER 8 (FULLY CONNECTED + SOFTMAX)
        fc8sm = all2all.All2AllSoftmax(
            self, name="fcsm8", output_shape=n_classes,
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=0)
        self._add_forward_unit(fc8sm)
        self._add_forward_unit(self.end_point)

    def initialize(self, device, **kwargs):
        learning_rate = kwargs.get("learning_rate")
        weights_decay = kwargs.get("weights_decay")
        super(Workflow, self).initialize(device, learning_rate=learning_rate,
                                         weights_decay=weights_decay)


def run(load, main):
    load(Workflow, layers=root.imagenet_caffe.layers)
    main(learning_rate=root.imagenet_caffe.learning_rate,
         weights_decay=root.imagenet_caffe.weights_decay)
