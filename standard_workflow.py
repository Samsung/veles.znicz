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
import veles.error as error


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
        self.layer_map = {
            "conv_tanh": (conv.ConvTanh,
                          gd_conv.GDTanhConv),
            "conv_relu": (conv.ConvRELU,
                          gd_conv.GDRELUConv),
            "conv_str": (conv.ConvStrictRELU,
                         gd_conv.GDStrictRELUConv),
            "conv": (conv.Conv,
                     gd_conv.GradientDescentConv),
            "norm": (normalization.LRNormalizerForward,
                     normalization.LRNormalizerBackward),
            "dropout": (dropout.DropoutForward,
                        dropout.DropoutBackward),
            "max_pooling": (pooling.MaxPooling,
                            gd_pooling.GDMaxPooling),
            "maxabs_pooling": (pooling.MaxAbsPooling,
                               gd_pooling.GDMaxAbsPooling),
            "avg_pooling": (pooling.AvgPooling,
                            gd_pooling.GDAvgPooling),
            "all2all_relu": (all2all.All2AllRELU,
                             gd.GDRELU),
            "all2all_tanh": (all2all.All2AllTanh,
                             gd.GDTanh),
            "softmax": (all2all.All2AllSoftmax,
                        gd.GDSM),
            "activation_str": (activation.ForwardStrictRELU,
                               activation.BackwardStrictRELU),
            "activation_log": (activation.ForwardLog,
                               activation.BackwardLog),
            "activation_sincos": (activation.ForwardSinCos,
                                  activation.BackwardSinCos)}

    def _get_layer_type_kwargs(self, layer):
        if type(layer) != dict:
            raise error.BadFormatError("layers should be a list of dicts")
        tpe = layer.get("type", "").strip()
        if not len(tpe):
            raise error.BadFormatError(
                "layer type should be non-empty string")
        if not tpe in self.layer_map:
            raise error.NotExistsError("Unknown layer type %s" % tpe)
        kwargs = dict(layer)
        del kwargs["type"]
        return tpe, kwargs

    def parse_forwards_from_config(self):
        """
        Parsing forward units from config.
        Adds a new fowrard unit to self.fwds, links it with previous fwd unit
        by link_from and link_attrs. If self.fwds is empty, links unit with
        self.loader
        """
        if type(self.layers) != list:
            raise error.BadFormatError("layers should be a list of dicts")
        del self.fwds[:]
        for i in range(len(self.layers)):
            layer = self.layers[i]
            tpe, kwargs = self._get_layer_type_kwargs(layer)
            unit = self.layer_map[tpe][0](self, **kwargs)
            self._add_forward_unit(unit)

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
        if type(self.layers) != list:
            raise error.BadFormatError("layers should be a list of dicts")
        del self.gds[:]
        self.gds.extend(None for _ in self.layers)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            tpe, kwargs = self._get_layer_type_kwargs(layer)

            # Check corresponding forward unit type
            if not isinstance(self.fwds[i], self.layer_map[tpe][0]):
                raise error.BadFormatError(
                    "Forward layer %s at position %d "
                    "is not an instance of %s" %
                    (repr(self.fwds[i]), i, repr(self.layer_map[tpe][0])))
            unit = self.layer_map[tpe][1](self, **kwargs)
            self.gds[i] = unit

            # Link attributes
            if i < len(self.fwds) - 1:
                # Averaged error function over a minibatch
                # only makes sense for the last layer
                if self.gds[i].error_function_averaged:
                    self.warning("error_function_averaged is set to True "
                                 "for the layer %d, but it makes sense only "
                                 "for the last layer usually.", i)
                self.gds[i].link_from(self.gds[i + 1])
                self.gds[i].link_attrs(self.gds[i + 1],
                                       ("err_output", "err_input"))
            else:
                self.gds[-1].link_from(self.snapshotter)
                self.gds[-1].link_attrs(self.evaluator, "err_output")

            attrs = []
            for attr in ("input", "output", "weights", "bias",
                         "input_offs", "mask"):
                if hasattr(self.fwds[i], attr):
                    attrs.append(attr)
            self.gds[i].link_attrs(self.fwds[i], *attrs)

            self.gds[i].gate_skip = self.decision.gd_skip
            self.gds[i].link_attrs(self.loader,
                                   ("batch_size", "minibatch_size"))

        # Disable error backpropagation on the first layer
        self.gds[0].need_err_input = False
        # Enable averaged error function over a minibatch for the last layer
        self.gds[-1].error_function_averaged = True
