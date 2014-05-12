"""
Created on May 5, 2014

Standard workflow class definition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root
import veles.error as error
from veles.znicz import nn_units
from veles.znicz import conv, pooling, all2all
from veles.znicz import gd, gd_conv, gd_pooling
from veles.znicz import normalization, dropout


class StandardWorkflow(nn_units.NNWorkflow):
    """
    A base class for standard workflows with forward and backward propagation.
    Is able to automatically create backward units by pre-created forward units
    """
    def __init__(self, workflow, **kwargs):
        self.layers = kwargs.get("layers")
        self.device = kwargs.get("device")
        kwargs["layers"] = self.layers
        kwargs["device"] = self.device
        super(StandardWorkflow, self).__init__(workflow, **kwargs)

    def parsing_forwards_from_congfig(self):
        """
        Parsing forward units from config.
        Adds a new fowrard unit to self.fwds, links it with previous fwd unit
        by link_from and link_attrs. If self.fwds is empty, links unit with
        self.loader
        """
        del self.fwds[:]
        for i in range(0, len(self.layers)):
            layer = self.layers[i]
            if layer["type"] == "conv":
                aa = conv.ConvTanh(self, n_kernels=layer["n_kernels"],
                                   kx=layer["kx"], ky=layer["ky"],
                                   weights_filling=root.conv.weights_filling,
                                   weights_stddev=root.conv.weights_stddev,
                                   sliding=layer.get("sliding", (1, 1, 1, 1)),
                                   padding=layer.get("padding", (0, 0, 0, 0)),
                                   device=self.device)
            elif layer["type"] == "conv_relu":
                aa = conv.ConvRELU(
                    self, n_kernels=layer["n_kernels"],
                    kx=layer["kx"], ky=layer["ky"],
                    sliding=layer.get("sliding", (1, 1, 1, 1)),
                    padding=layer.get("padding", (0, 0, 0, 0)),
                    device=self.device,
                    weights_filling=root.conv_relu.weights_filling,
                    weights_stddev=root.conv_relu.weights_stddev,)
            elif layer["type"] == "max_pooling":
                aa = pooling.MaxPooling(self, kx=layer["kx"], ky=layer["ky"],
                                        sliding=layer.get("sliding",
                                                          (layer["kx"],
                                                           layer["ky"])),
                                        device=self.device)
            elif layer["type"] == "avg_pooling":
                aa = pooling.AvgPooling(self, kx=layer["kx"], ky=layer["ky"],
                                        sliding=layer.get("sliding",
                                                          (layer["kx"],
                                                           layer["ky"])),
                                        device=self.device)
            elif layer["type"] == "all2all_relu":
                aa = all2all.All2AllRELU(
                    self, output_shape=[layer["layers"]], device=self.device,
                    weights_filling=root.all2all_relu.weights_filling,
                    weights_stddev=root.all2all_relu.weights_stddev)
            elif layer["type"] == "all2all_tanh":
                aa = all2all.All2AllTanh(
                    self, output_shape=[layer["layers"]], device=self.device,
                    weights_filling=root.all2all_tanh.weights_filling,
                    weights_stddev=root.all2all_tanh.weights_stddev)
            elif layer["type"] == "softmax":
                aa = all2all.All2AllSoftmax(
                    self, output_shape=[layer["layers"]], device=self.device,
                    weights_filling=root.softmax.weights_filling,
                    weights_stddev=root.softmax.weights_stddev)
            else:
                raise error.ErrBadFormat("Unsupported layer type %s" %
                                         (layer["type"]))

            self.fwds.append(aa)
            if i:
                self.fwds[-1].link_from(self.fwds[-2])
                self.fwds[-1].link_attrs(self.fwds[-2],
                                         ("input", "output"))
            else:
                self.fwds[-1].link_from(self.loader)
                self.fwds[-1].link_attrs(self.loader,
                                         ("input", "minibatch_data"))

    def _create_gradient_descent_units(self):
        '''
        Creates gradient descent units for previously made self.fwds.
        Feeds their inputs with respect of their order.
        '''
        self.gds = []
        for i, fwd_elm in enumerate(self.fwds):
            if isinstance(fwd_elm, conv.ConvRELU):
                grad_elm = gd_conv.GDRELUConv(
                    self, n_kernels=fwd_elm.n_kernels, kx=fwd_elm.kx,
                    ky=fwd_elm.ky, sliding=fwd_elm.sliding,
                    padding=fwd_elm.padding, device=self.device)

            elif isinstance(fwd_elm, conv.ConvTanh):
                grad_elm = gd_conv.GDTanhConv(
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

            elif isinstance(fwd_elm, dropout.DropoutForward):
                grad_elm = dropout.DropoutBackward(
                    self, dropout_ratio=fwd_elm.dropout_ratio,
                    device=self.device)

            else:
                raise ValueError("Unsupported unit type " + str(type(fwd_elm)))

            self.gds.append(grad_elm)

            grad_elm.link_attrs(fwd_elm, ("y", "output"), ("h", "input"))
            grad_elm.link_attrs(self.loader, ("batch_size", "minibatch_size"))

            # LRN has no weights
            if not isinstance(grad_elm, normalization.LRNormalizerBackward):
                grad_elm.link_attrs(fwd_elm, "weights")

            # LRN and droupout units have no biases
            if not (isinstance(grad_elm, dropout.DropoutBackward) or
                    isinstance(grad_elm, normalization.LRNormalizerBackward)):
                grad_elm.link_attrs(fwd_elm, "bias")

            grad_elm.gate_skip = self.decision.gd_skip

        for i in range(len(self.gds) - 1):
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.gds[i + 1], ("err_y", "err_h"))

        self.gds[-1].link_from(self.decision)
        self.gds[-1].link_attrs(self.evaluator, "err_y")
