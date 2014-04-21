#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on Apr 18, 2014

@author: Alexey Golovizin <a.golovizin@samsung.com>
"""

from veles.config import root, get
from veles.znicz import nn_units
from veles.znicz import loader
from veles.znicz import conv, pooling, all2all, evaluator, decision
from veles.znicz import gd, gd_conv, gd_pooling


class Loader(loader.FullBatchLoader):
    pass


class Workflow(nn_units.NNWorkflow):
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)
        self.device = device

        self.rpt.link_from(self.start_point)

        self.loader = Loader(self)
        self.loader.link_from(self.rpt)

        # FORWARD LAYERS

        # Layer 1 (CONV + POOL)
        conv1 = conv.ConvRELU(
            self, n_kernels=96, kx=11, ky=11,
            sliding=(4, 4, 4, 4), padding=(0, 0, 0, 0), device=device)
        conv1.link_from(self.loader)
        conv1.link_attrs(self.loader, ("input", "minibatch_data"))
        self.forward.append(conv1)

        pool1 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
                                   device=device)
        self.link_last_forward_with(pool1)

        # TODO: normalization, gaussian filling

        # Layer 2 (CONV + POOL)
        conv2 = conv.ConvRELU(
            self, n_kernels=256, kx=5, ky=5,
            sliding=(1, 1, 1, 1), padding=(2, 2, 2, 2), device=device)
        self.link_last_forward_with(conv2)

        pool2 = pooling.MaxPooling(
            self, kx=3, ky=3, sliding=(2, 2), device=device)
        self.add_to_forward_unts(pool2)

        # Layer 3 (CONV)
        conv3 = conv.ConvRELU(
            self, n_kernels=384, kx=3, ky=3,
            sliding=(1, 1, 1, 1), padding=(1, 1, 1, 1), device=device)
        self.link_last_forward_with(conv3)

        # Layer 4 (CONV)
        conv4 = conv.ConvRELU(
            self, n_kernels=384, kx=3, ky=3,
            sliding=(1, 1, 1, 1), padding=(1, 1, 1, 1), device=device)
        self.link_last_forward_with(conv4)

        # Layer 5 (CONV + POOL)
        conv5 = conv.ConvRELU(
            self, n_kernels=256, kx=3, ky=3,
            sliding=(1, 1, 1, 1), padding=(1, 1, 1, 1), device=device)
        self.link_last_forward_with(conv5)

        pool5 = pooling.MaxPooling(
            self, kx=3, ky=3, sliding=(2, 2),
            device=device)
        self.link_last_forward_with(pool5)

        # Layer 6 (FULLY CONNECTED)
        fc6 = all2all.All2AllRELU(self, output_shape=4096)
        self.link_last_forward_with(fc6)

            # TODO: dropout

        # Layer 7 (FULLY CONNECTED)
        fc7 = all2all.All2AllRELU(self, output_shape=4096)
        self.link_last_forward_with(fc7)

            # TODO: dropout

        # LAYER 8 (FULLY CONNECTED + SOFTMAX)
        fc8sm = all2all.All2AllSoftmax(self, output_shape=4096)
        self.link_last_forward_with(fc8sm)
        self.forward.append(fc8sm)

        # EVALUATOR
        self.ev = evaluator.EvaluatorSoftmax(self, device=device)
        self.ev.link_from(self.fc8)
        self.ev.y = self.fc8.output
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_idx = fc8sm.max_idx
        self.ev.link_attrs(self.loader, ("batch_size", "minibatch_size"),
                           ("max_samples_per_epoch", "total_samples"))

        # DECISION UNIT
        self.decision = decision.Decision(self)
        self.decision.link_from(self.ev)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class", "minibatch_last")
        self.decision.link_attrs(
            self.ev, ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_y_sum"))
        self.decision.class_samples = self.loader.class_samples

        # BACKWARD LAYERS (GRADIENT DESCENT)
        self._create_gradient_descent_units()

        # Repeater from last-activated GD unit
        self.rpt.link_from(self.gd[0])

    def initialize(self, global_alpha, global_lambda, minibatch_maxsize):
        return super(Workflow, self).initialize()

    def _create_gradient_descent_units(self):
        '''
        Creates gradient descent units for previously made self.forward.
        Feeds their inputs with respect of their order.
        '''
        self.gd = []
        for i, fwd_elm in enumerate(self.forward):
            if isinstance(fwd_elm, conv.ConvRELU):
                grad_elm = gd_conv.GDRELU(
                    self, n_kernels=fwd_elm.n_kernels,
                    kx=fwd_elm.kx, ky=fwd_elm.ky, sliding=fwd_elm.sliding,
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
                    self, kx=fwd_elm.kx,
                    ky=fwd_elm.ky, sliding=fwd_elm.sliding, device=self.device)
                grad_elm.link_attrs(fwd_elm, ("h_offs", "input_offs"))  # WHY?!

            elif isinstance(fwd_elm, pooling.AvgPooling):
                grad_elm = gd_pooling.GDAvgPooling(
                    self, kx=fwd_elm.kx,
                    ky=fwd_elm.ky, sliding=fwd_elm.sliding, device=self.device)

            else:
                raise ValueError("Unsupported unit type " + str(type(fwd_elm)))

            self.gd.append(grad_elm)

            grad_elm.link_attrs(
                fwd_elm, ("y", "output"), ("h", "input"), "weights", "bias")
            grad_elm.gate_skip = self.decision.gd_skip

        for i in range(len(self.gd) - 1):
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].link_attrs(self.gd[i + 1], ("err_y", "err_h"))

        self.gd[-1].link_from(self.decision)
        self.gd[-1].link_attrs(self.ev, "err_y")

    def link_last_forward_with(self, new_unit):
        '''
        Adds a new forward unit to self.forward, links it with previous unit
        by link_from and link_attrs. If self.forward is empty, raises error.
        '''
        assert self.forward
        prev_forward_unit = self.forward[-1]
        new_unit.link_from(prev_forward_unit)
        new_unit.link_attrs(prev_forward_unit, ("input", "output"))


def run(load, main):
    load(Workflow, layers=root.imagenet.layers)
    main(global_alpha=get(root.imagenet.global_alpha),
         global_lambda=root.imagenet.global_lambda,
         minibatch_maxsize=root.loader.minibatch_maxsize)
