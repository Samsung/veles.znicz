#!/usr/bin/python3.3 -O
# encoding: utf-8
"""
Created on Apr 18, 2014

This workflow should clone the Imagenet example in CAFFE tool with deconv.
"""

from veles.config import root
from veles.znicz import conv, pooling
from veles.znicz import normalization
from veles.znicz.standard_workflow import StandardWorkflow
from veles.znicz.depooling import Depooling
from veles.znicz.tests.research.kernels.imagenet_loader\
    import ImgLoaderClassifier
from veles.znicz.deconv import Deconv

import logging

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
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        kwargs["name"] = kwargs.get("name", "ImageNet")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.loader = ImgLoaderClassifier(self, path_mean_img="./data/"
                                          "ilsvrc_2012_mean.npy",
                                          max_minibatch_size=3,
                                          minibatch_size=3,
                                          test_paths=["./data/test"],
                                          validation_paths=[],
                                          train_paths=[])
        self.loader.setup(level=logging.DEBUG)

        self.loader.load_data()

        self.loader.link_from(self.start_point)

        conv1 = conv.ConvStrictRELU(
            self, name="conv1",
            n_kernels=96, kx=11, ky=11, sliding=(4, 4), padding=(0, 0, 0, 0),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=0)
        self._add_forward_unit(conv1)

        pool1 = pooling.MaxPooling(self, name="pool1",
                                   kx=3, ky=3, sliding=(2, 2))
        self._add_forward_unit(pool1)

        norm1 = normalization.LRNormalizerForward(
            self, name="norm1", alpha=0.00002, beta=0.75, n=5, k=1)
        self._add_forward_unit(norm1)

        # Layer 2 (CONV + POOL)
        conv2 = conv.ConvStrictRELU(
            self, name="conv2",
            n_kernels=256, kx=5, ky=5, sliding=(1, 1), padding=(2, 2, 2, 2),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=1.0)
        self._add_forward_unit(conv2)

        pool2 = pooling.MaxPooling(self, name="pool2",
                                   kx=3, ky=3, sliding=(2, 2))
        self._add_forward_unit(pool2)

        norm2 = normalization.LRNormalizerForward(
            self, name="norm2", alpha=0.00002, beta=0.75, n=5, k=1,)
        self._add_forward_unit(norm2)

        # Layer 3 (CONV)
        conv3 = conv.ConvStrictRELU(
            self, name="conv3",
            n_kernels=384, kx=3, ky=3, sliding=(1, 1), padding=(1, 1, 1, 1),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=0)
        self._add_forward_unit(conv3)

        # Layer 4 (CONV)
        conv4 = conv.ConvStrictRELU(
            self, name="conv4",
            n_kernels=384, kx=3, ky=3, sliding=(1, 1), padding=(1, 1, 1, 1),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=1)
        self._add_forward_unit(conv4)

        # Layer 5 (CONV + POOL)
        conv5 = conv.ConvStrictRELU(
            self, name="conv5",
            n_kernels=256, kx=3, ky=3, sliding=(1, 1), padding=(1, 1, 1, 1),
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=1)
        self._add_forward_unit(conv5)

        pool5 = pooling.MaxPooling(self, name="pool5",
                                   kx=3, ky=3, sliding=(2, 2))
        self._add_forward_unit(pool5)

        #Layer -5 (DECONV + DEPOOL)
        depool5 = Depooling(self, name="depool5", sliding=pool5.sliding)
        depool5.link_from(pool5)

        deconv5 = Deconv(
            self, name="deconv5", n_kernels=conv5.n_kernels, kx=conv5.kx,
            ky=conv5.ky, sliding=conv2.sliding, padding=conv2.padding,
            unsafe_padding=True)
        deconv5.link_from(depool5)

        #Layer -4 (DECONV)

        deconv4 = Deconv(
            self, name="deconv4", n_kernels=conv4.n_kernels, kx=conv4.kx,
            ky=conv4.ky, sliding=conv4.sliding, padding=conv4.padding,
            unsafe_padding=True)
        deconv4.link_from(deconv5)

        #Layer -3 (DECONV)

        deconv3 = Deconv(self, name="deconv3", n_kernels=conv3.n_kernels,
                         kx=conv3.kx, ky=conv3.ky, sliding=conv3.sliding,
                         padding=conv3.padding, unsafe_padding=True)
        deconv3.link_from(deconv4)

        #Layer -2 (DECONV + POOL)
        depool2 = Depooling(self, name="depool2", sliding=pool2.sliding)
        depool2.link_from(deconv3)

        deconv2 = Deconv(self, name="deconv2", n_kernels=conv2.n_kernels,
                         kx=conv2.kx, ky=conv2.ky, sliding=conv2.sliding,
                         padding=conv2.padding, unsafe_padding=True)
        deconv2.link_from(depool2)

        depool1 = Depooling(self, name="depool1", sliding=pool1.sliding)
        depool1.link_from(deconv2)

        deconv1 = Deconv(self, name="deconv1", n_kernels=conv1.n_kernels,
                         kx=conv1.kx, ky=conv1.ky, sliding=conv1.sliding,
                         padding=conv1.padding, unsafe_padding=True)
        deconv1.link_from(depool1)

        self.end_point.link_from(deconv1)

    def initialize(self, device, **kwargs):
        learning_rate = kwargs.get("learning_rate")
        weights_decay = kwargs.get("weights_decay")
        super(Workflow, self).initialize(device, learning_rate=learning_rate,
                                         weights_decay=weights_decay)


def run(load, main):
    load(Workflow, layers=root.imagenet_caffe.layers)
    main(learning_rate=root.imagenet_caffe.learning_rate,
         weights_decay=root.imagenet_caffe.weights_decay)
