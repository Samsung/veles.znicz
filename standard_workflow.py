"""
Created on May 5, 2014

Standard workflow class definition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.znicz import nn_units
from veles.znicz import conv, pooling, all2all
from veles.znicz import gd, gd_conv, gd_pooling
from veles.znicz import normalization, dropout
from veles.znicz import activation


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

    def parse_forwards_from_config(self):
        """
        Parsing forward units from config.
        Adds a new fowrard unit to self.fwds, links it with previous fwd unit
        by link_from and link_attrs. If self.fwds is empty, links unit with
        self.loader
        """
        del self.fwds[:]
        for i in range(0, len(self.layers)):
            layer = self.layers[i]
            kwargs = {"weights_filling": layer.get("weights_filling",
                                                   "uniform"),
                      "weights_stddev": layer.get("weights_stddev"),
                      "bias_filling": layer.get("bias_filling",
                                                "uniform"),
                      "bias_stddev": layer.get("bias_stddev")}
            layer_ct = {"conv_tanh": lambda layer:
                        conv.ConvTanh(
                            self, n_kernels=layer["n_kernels"],
                            kx=layer["kx"], ky=layer["ky"],
                            sliding=layer.get("sliding", (1, 1, 1, 1)),
                            padding=layer.get("padding", (0, 0, 0, 0)),
                            device=self.device, **kwargs),
                        "conv_relu": lambda layer:
                        conv.ConvRELU(
                            self, n_kernels=layer["n_kernels"],
                            kx=layer["kx"], ky=layer["ky"],
                            sliding=layer.get("sliding", (1, 1, 1, 1)),
                            padding=layer.get("padding", (0, 0, 0, 0)),
                            device=self.device, **kwargs),
                        "conv_str": lambda layer:
                        conv.ConvStrictRELU(
                            self, n_kernels=layer["n_kernels"],
                            kx=layer["kx"], ky=layer["ky"],
                            sliding=layer.get("sliding", (1, 1, 1, 1)),
                            padding=layer.get("padding", (0, 0, 0, 0)),
                            device=self.device, **kwargs),
                        "conv": lambda layer:
                        conv.Conv(
                            self, n_kernels=layer["n_kernels"],
                            kx=layer["kx"], ky=layer["ky"],
                            sliding=layer.get("sliding", (1, 1, 1, 1)),
                            padding=layer.get("padding", (0, 0, 0, 0)),
                            device=self.device, **kwargs),
                        "norm": lambda layer:
                        normalization.LRNormalizerForward(
                            self, alpha=layer.get("alpha", (0.00005)),
                            beta=layer.get("beta", (0.75)),
                            n=layer.get("n", (3)),
                            device=self.device),
                        "dropout": lambda layer:
                        dropout.DropoutForward(
                            self,
                            dropout_ratio=layer.get("dropout_ratio", (0.5)),
                            device=self.device),
                        "max_pooling": lambda layer:
                        pooling.MaxPooling(
                            self, kx=layer["kx"], ky=layer["ky"],
                            sliding=layer.get("sliding", (layer["kx"],
                                                          layer["ky"])),
                            device=self.device, **kwargs),
                        "avg_pooling": lambda layer:
                        pooling.AvgPooling(
                            self, kx=layer["kx"], ky=layer["ky"],
                            sliding=layer.get("sliding", (layer["kx"],
                                                          layer["ky"])),
                            device=self.device, **kwargs),
                        "all2all_relu": lambda layer:
                        all2all.All2AllRELU(
                            self,
                            output_shape=self._as_list(layer["output_shape"]),
                            device=self.device, **kwargs),
                        "all2all_tanh": lambda layer:
                        all2all.All2AllTanh(
                            self,
                            output_shape=self._as_list(layer["output_shape"]),
                            device=self.device, **kwargs),
                        "softmax": lambda layer:
                        all2all.All2AllSoftmax(
                            self,
                            output_shape=self._as_list(layer["output_shape"]),
                            device=self.device, **kwargs),
                        "activation_str": lambda layer:
                        activation.ForwardStrictRELU(
                            self, device=self.device, **kwargs),
                        "activation_log": lambda layer:
                        activation.ForwardLog(
                            self, device=self.device, **kwargs),
                        "activation_sincos": lambda layer:
                        activation.ForwardSinCos(
                            self, device=self.device, **kwargs)}

            unit = layer_ct[layer["type"]](layer)
            self._add_forward_unit(unit)

    def _as_list(self, vle):
        if type(vle) in (int, float, str):
            return [vle]
        return vle

    def _add_forward_unit(self, new_unit):
        """
        Adds a new fowrard unit to self.fwds, links it with previous fwd unit
        by link_from and link_attrs. If self.fwds is empty, links unit with
        self.loader
        """
        if len(self.fwds):
            prev_forward_unit = self.fwds[-1]
            new_unit.link_attrs(prev_forward_unit, ("input", "output"))
        else:
            assert self.loader is not None
            prev_forward_unit = self.loader
            new_unit.link_attrs(self.loader, ("input", "minibatch_data"))

        new_unit.link_from(prev_forward_unit)
        self.fwds.append(new_unit)

    def create_gradient_descent_units(self):
        """
        Creates gradient descent units for previously made self.fwds.
        Feeds their inputs with respect of their order.
        """
        self.gds = []
        for i, fwd_elm in enumerate(self.fwds):
            layer = self.layers[i]
            kwargs = {}
            for name in ("learning_rate", "weights_decay", "gradient_moment",
                         "learning_rate_bias", "weights_decay_bias",
                         "gradient_moment_bias"):
                if name in layer:
                    kwargs[name] = layer[name]
            if isinstance(fwd_elm, conv.ConvRELU):
                grad_elm = gd_conv.GDRELUConv(
                    self, n_kernels=fwd_elm.n_kernels, kx=fwd_elm.kx,
                    ky=fwd_elm.ky, sliding=fwd_elm.sliding,
                    padding=fwd_elm.padding, device=self.device, **kwargs)

            elif isinstance(fwd_elm, conv.Conv):
                grad_elm = gd_conv.GradientDescentConv(
                    self, n_kernels=fwd_elm.n_kernels,
                    kx=fwd_elm.kx, ky=fwd_elm.ky, sliding=fwd_elm.sliding,
                    padding=fwd_elm.padding, device=self.device, **kwargs)

            elif isinstance(fwd_elm, conv.ConvTanh):
                grad_elm = gd_conv.GDTanhConv(
                    self, n_kernels=fwd_elm.n_kernels,
                    kx=fwd_elm.kx, ky=fwd_elm.ky, sliding=fwd_elm.sliding,
                    padding=fwd_elm.padding, device=self.device, **kwargs)

            elif isinstance(fwd_elm, conv.ConvStrictRELU):
                grad_elm = gd_conv.GDStrictRELUConv(
                    self, n_kernels=fwd_elm.n_kernels,
                    kx=fwd_elm.kx, ky=fwd_elm.ky, sliding=fwd_elm.sliding,
                    padding=fwd_elm.padding, device=self.device, **kwargs)

            elif isinstance(fwd_elm, all2all.All2AllRELU):
                grad_elm = gd.GDRELU(self, device=self.device, **kwargs)

            elif isinstance(fwd_elm, all2all.All2AllSoftmax):
                grad_elm = gd.GDSM(self, device=self.device, **kwargs)

            elif isinstance(fwd_elm, all2all.All2AllTanh):
                grad_elm = gd.GDTanh(self, device=self.device, **kwargs)

            elif isinstance(fwd_elm, pooling.MaxPooling):
                grad_elm = gd_pooling.GDMaxPooling(
                    self, kx=fwd_elm.kx, ky=fwd_elm.ky,
                    sliding=fwd_elm.sliding,
                    device=self.device, **kwargs)
                grad_elm.link_attrs(fwd_elm, "input_offs")

            elif isinstance(fwd_elm, pooling.AvgPooling):
                grad_elm = gd_pooling.GDAvgPooling(
                    self, kx=fwd_elm.kx, ky=fwd_elm.ky,
                    sliding=fwd_elm.sliding,
                    device=self.device, **kwargs)

            elif isinstance(fwd_elm, normalization.LRNormalizerForward):
                grad_elm = normalization.LRNormalizerBackward(
                    self, alpha=fwd_elm.alpha, beta=fwd_elm.beta,
                    k=fwd_elm.k, n=fwd_elm.n, **kwargs)

            elif isinstance(fwd_elm, dropout.DropoutForward):
                grad_elm = dropout.DropoutBackward(
                    self, dropout_ratio=fwd_elm.dropout_ratio,
                    device=self.device, **kwargs)

            elif isinstance(fwd_elm, activation.ForwardStrictRELU):
                grad_elm = activation.BackwardStrictRELU(
                    self, device=self.device, **kwargs)

            elif isinstance(fwd_elm, activation.ForwardLog):
                grad_elm = activation.BackwardLog(
                    self, device=self.device, **kwargs)

            elif isinstance(fwd_elm, activation.ForwardSinCos):
                grad_elm = activation.BackwardSinCos(
                    self, device=self.device, **kwargs)

            else:
                raise ValueError("Unsupported unit type " + str(type(fwd_elm)))

            self.gds.append(grad_elm)

        for i in range(len(self.fwds) - 1, -1, -1):
            if i < len(self.fwds) - 1:
                self.gds[i].link_from(self.gds[i + 1])
                self.gds[i].link_attrs(self.gds[i + 1],
                                       ("err_output", "err_input"))
            else:
                self.gds[-1].link_from(self.snapshotter)
                self.gds[-1].link_attrs(self.evaluator, "err_output")

            self.gds[i].link_attrs(self.fwds[i], "input", "output",
                                   "weights", "bias")
            self.gds[i].gate_skip = self.decision.gd_skip
            self.gds[i].link_attrs(self.loader,
                                   ("batch_size", "minibatch_size"))
        self.gds[0].need_err_input = False
