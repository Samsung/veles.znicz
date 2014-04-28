#!/usr/bin/python3.3 -O
# encoding: utf-8
"""
Created on Apr 18, 2014

@author: Alexey Golovizin <a.golovizin@samsung.com>
"""

from veles.config import root
from veles.znicz import nn_units
from veles.znicz import conv, pooling, all2all, evaluator, decision
from veles.znicz import gd, gd_conv, gd_pooling
from veles.znicz import normalization
from veles.znicz.samples.imagenet.loader import LoaderDetection


import logging

root.defaults = {"all2all": {"weights_magnitude": 0.05},
                 "decision": {"fail_iterations": 100,
                              "snapshot_prefix": "imagenet_caffe",
                              "store_samples_mse": True},
                 "loader": {"minibatch_maxsize": 60},
                 "imagenet_caffe": {"global_alpha": 0.01,
                                    "global_lambda": 0.0}}


class Workflow(nn_units.NNWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        kwargs["name"] = kwargs.get("name", "MNIST")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = LoaderDetection(self,
                                      ipath="/data/imagenet/2013",
                                      dbpath="/data/imagenet/2013/db",
                                      year="2013", series="DET")
        self.loader.setup(level=logging.DEBUG)
        self.loader.load_data()

        self.loader.link_from(self.repeater)

        # FORWARD LAYERS

        # Layer 1 (CONV + POOL)
        conv1 = conv.ConvRELU(self, n_kernels=96, kx=11, ky=11,
                              sliding=(4, 4), padding=(0, 0, 0, 0),
                              device=device)
        conv1.link_from(self.loader)
        conv1.link_attrs(self.loader, ("input", "minibatch_data"))
        self.forward.append(conv1)

        pool1 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
                                   device=device)
        self._link_last_forward_with(pool1)

        # TODO: normalization, gaussian filling
        norm1 = normalization.LRNormalizerForward(self, device=device)
        self._link_last_forward_with(norm1)

        # Layer 2 (CONV + POOL)
        conv2 = conv.ConvRELU(self, n_kernels=256, kx=5, ky=5,
                              sliding=(1, 1), padding=(2, 2, 2, 2),
                              device=device)
        self._link_last_forward_with(conv2)

        pool2 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
                                   device=device)
        self._link_last_forward_with(pool2)

        # Layer 3 (CONV)
        conv3 = conv.ConvRELU(self, n_kernels=384, kx=3, ky=3,
                              sliding=(1, 1), padding=(1, 1, 1, 1),
                              device=device)
        self._link_last_forward_with(conv3)

        # Layer 4 (CONV)
        conv4 = conv.ConvRELU(self, n_kernels=384, kx=3, ky=3,
                              sliding=(1, 1), padding=(1, 1, 1, 1),
                              device=device)
        self._link_last_forward_with(conv4)

        # Layer 5 (CONV + POOL)
        conv5 = conv.ConvRELU(self, n_kernels=256, kx=3, ky=3,
                              sliding=(1, 1), padding=(1, 1, 1, 1),
                              device=device)
        self._link_last_forward_with(conv5)

        pool5 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
                                   device=device)
        self._link_last_forward_with(pool5)

        # Layer 6 (FULLY CONNECTED)
        fc6 = all2all.All2AllRELU(self, output_shape=4096, device=device)
        self._link_last_forward_with(fc6)

            # TODO: dropout

        # Layer 7 (FULLY CONNECTED)
        fc7 = all2all.All2AllRELU(self, output_shape=4096, device=device)
        self._link_last_forward_with(fc7)

            # TODO: dropout

        # LAYER 8 (FULLY CONNECTED + SOFTMAX)
        fc8sm = all2all.All2AllSoftmax(self, output_shape=1000,
                                       device=device)
        self._link_last_forward_with(fc8sm)

        # EVALUATOR
        self.ev = evaluator.EvaluatorSoftmax(self, device=device)
        self.ev.link_from(fc8sm)
        self.ev.link_attrs(fc8sm, ("y", "output"), "max_idx")
        self.ev.link_attrs(self.loader, ("batch_size", "minibatch_size"),
                           ("max_samples_per_epoch", "total_samples"),
                           ("labels", "minibatch_labels"))

        # Add decision unit
        self.decision = decision.Decision(self)
        self.decision.link_from(self.ev)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "class_samples",
                                 "no_more_minibatches_left")
        self.decision.link_attrs(
            self.ev,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_y_sum"))

        # BACKWARD LAYERS (GRADIENT DESCENT)
        self._create_gradient_descent_units()

        # repeater and gate block
        self.repeater.link_from(self.gd[0])
        self.end_point.link_from(self.gd[0])
        self.end_point.gate_block = ~self.decision.complete
        self.loader.gate_block = self.decision.complete

    def _create_gradient_descent_units(self):
        '''
        Creates gradient descent units for previously made self.forward.
        Feeds their inputs with respect of their order.
        '''
        self.gd = []
        for i, fwd_elm in enumerate(self.forward):
            if isinstance(fwd_elm, conv.ConvRELU):
                grad_elm = gd_conv.GDRELU(
                    self, n_kernels=fwd_elm.n_kernels, kx=fwd_elm.kx,
                    ky=fwd_elm.ky, sliding=fwd_elm.sliding,
                    padding=fwd_elm.padding, device=self.device)

            elif isinstance(fwd_elm, conv.ConvTanh):
                grad_elm = gd_conv.GDTanh(
                    self, n_kernels=fwd_elm.n_kernels,
                    kx=fwd_elm.kx, ky=fwd_elm.ky, sliding=fwd_elm.sliding,
                    padding=fwd_elm.padding, device=self.device)

            elif isinstance(fwd_elm, all2all.All2AllRELU):
                grad_elm = gd.GDRELU(self, device=self.device)

            elif isinstance(fwd_elm, all2all.All2AllSoftmax):
                grad_elm = gd.GDSM(self, device=self.device)

            elif isinstance(fwd_elm, all2all.All2AllTanh):
                grad_elm = gd.GDTanh(self, device=self.device)

            elif isinstance(fwd_elm, pooling.MaxPooling):
                grad_elm = gd_pooling.GDMaxPooling(
                    self, kx=fwd_elm.kx, ky=fwd_elm.ky,
                    sliding=fwd_elm.sliding,
                    device=self.device)
                grad_elm.link_attrs(fwd_elm, ("h_offs", "input_offs"))  # WHY?!

            elif isinstance(fwd_elm, pooling.AvgPooling):
                grad_elm = gd_pooling.GDAvgPooling(
                    self, kx=fwd_elm.kx, ky=fwd_elm.ky,
                    sliding=fwd_elm.sliding,
                    device=self.device)

            elif isinstance(fwd_elm, normalization.LRNormalizerForward):
                grad_elm = normalization.LRNormalizerBackward(
                    self, alpha=fwd_elm.alpha, beta=fwd_elm.beta,
                    k=fwd_elm.k, n=fwd_elm.n)

            else:
                raise ValueError("Unsupported unit type " + str(type(fwd_elm)))

            self.gd.append(grad_elm)

            grad_elm.link_attrs(fwd_elm, ("y", "output"), ("h", "input"),
                                "weights", "bias")
            grad_elm.gate_skip = self.decision.gd_skip

        for i in range(len(self.gd) - 1):
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].link_attrs(self.gd[i + 1], ("err_y", "err_h"))

        self.gd[-1].link_from(self.decision)
        self.gd[-1].link_attrs(self.ev, "err_y")

    def _link_last_forward_with(self, new_unit):
        '''
        Adds a new forward unit to self.forward, links it with previous unit
        by link_from and link_attrs. If self.forward is empty, raises error.
        '''
        assert self.forward
        prev_forward_unit = self.forward[-1]
        new_unit.link_from(prev_forward_unit)
        new_unit.link_attrs(prev_forward_unit, ("input", "output"))
        self.forward.append(new_unit)

    def initialize(self, global_alpha, global_lambda, device):
        super(Workflow, self).initialize(global_alpha=global_alpha,
                                         global_lambda=global_lambda,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.imagenet_caffe.layers)
    main(global_alpha=root.imagenet_caffe.global_alpha,
         global_lambda=root.imagenet_caffe.global_lambda)
