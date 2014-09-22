#!/usr/bin/python3 -O
# encoding: utf-8
"""
Created on Apr 18, 2014

This workflow should clone the Imagenet example in CAFFE tool.
"""

from veles.config import root
from veles.znicz import conv, pooling, all2all, evaluator, decision
from veles.znicz import normalization, dropout, activation
from veles.znicz.nn_units import NNSnapshotter
from veles.znicz.tests.research.imagenet.loader import LoaderDetection
from veles.znicz.standard_workflow import StandardWorkflow
from veles.znicz import lr_adjust

root.common.update = {"precision_type": "float", }

root.defaults = {
    "precision_type": "float",
    "decision": {"fail_iterations": 100,
                 "store_samples_mse": True},
    "snapshotter": {"prefix": "imagenet_caffe"},
    "loader": {"minibatch_size": 256},
    "imagenet_caffe": {"learning_rate": 0.0001,
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
        kwargs["name"] = kwargs.get("name", "ImageNet")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = LoaderDetection(
            self, max_minibatch_size=256, minibatch_size=128,
            ipath="/data/veles/datasets/imagenet/2013",
            dbpath="/data/veles/datasets/imagenets/2013/db",
            year="2013", series="img")
        self.loader.load_data()

        self.loader.link_from(self.repeater)

        # FORWARD LAYERS

        # Layer 1 (CONV + POOL)
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
            self, name="norm1", alpha=0.001, beta=0.75, n=5, k=1)
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
            self, name="norm2", alpha=0.001, beta=0.75, n=5, k=1,)
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

        # Layer 6 (FULLY CONNECTED + 50% dropout)
        fc6 = all2all.All2All(
            self, name="fc6",
            output_shape=4096,
            weights_filling="gaussian", weights_stddev=0.005,
            bias_filling="constant", bias_stddev=1)
        self._add_forward_unit(fc6)

        relu6 = activation.ForwardStrictRELU(self, name="relu6")
        self._add_forward_unit(relu6)

        drop6 = dropout.DropoutForward(self, name="drop6", dropout_ratio=0.5)
        self._add_forward_unit(drop6)

        # Layer 7 (FULLY CONNECTED + 50% dropout)
        fc7 = all2all.All2All(
            self, name="fc7", output_shape=4096, weights_filling="gaussian",
            weights_stddev=0.005)
        self._add_forward_unit(fc7)

        relu7 = activation.ForwardStrictRELU(self, name="relu7")
        self._add_forward_unit(relu7)

        drop7 = dropout.DropoutForward(self, name="drop7", dropout_ratio=0.5)
        self._add_forward_unit(drop7)

        # LAYER 8 (FULLY CONNECTED + SOFTMAX)
        fc8sm = all2all.All2AllSoftmax(
            self, name="fcsm8", output_shape=n_classes,
            weights_filling="gaussian", weights_stddev=0.01,
            bias_filling="constant", bias_stddev=0)
        self._add_forward_unit(fc8sm)

        # EVALUATOR
        self.evaluator = evaluator.EvaluatorSoftmax(self, name="eval")
        self.evaluator.link_from(fc8sm)
        self.evaluator.link_attrs(fc8sm, "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("labels", "minibatch_labels"))

        # Add decision unit
        self.decision = decision.DecisionGD(self)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class", "minibatch_size",
                                 "last_minibatch", "class_lengths",
                                 "epoch_ended", "epoch_number")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_output_sum"))

        self.snapshotter = NNSnapshotter(self, prefix=root.snapshotter.prefix,
                                         directory=root.common.snapshot_dir)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.decision.epoch_ended | ~self.decision.improved)

        # BACKWARD LAYERS (GRADIENT DESCENT)

        lr_policy = lr_adjust.StepExpPolicy(0.01, 0.1, 100000)
        bias_lr_policy = lr_adjust.StepExpPolicy(0.02, 0.1, 100000)
        lr_adjuster = lr_adjust.LearningRateAdjust(self)

        self.create_gd_units(
            lr_adjuster, learning_rate=0.01, learning_rate_bias=0.02,
            gradient_moment=0.9, weights_decay=0.0005, weights_decay_bias=0)

        # repeater and gate block
        self.repeater.link_from(self.gds[0])
        self.end_point.link_from(self.snapshotter)
        self.end_point.gate_block = ~self.decision.complete
        self.loader.gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device, **kwargs):
        super(Workflow, self).initialize(learning_rate=learning_rate,
                                         weights_decay=weights_decay,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.imagenet_caffe.layers)
    main(learning_rate=root.imagenet_caffe.learning_rate,
         weights_decay=root.imagenet_caffe.weights_decay)
