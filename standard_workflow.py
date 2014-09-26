"""
Created on May 5, 2014

Standard workflow class definition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import sys

if sys.version_info > (3, 0):
    from collections import UserDict
else:
    from UserDict import UserDict

from veles.znicz import nn_units
from veles.znicz import conv, pooling, all2all
from veles.znicz import gd, gd_conv, gd_pooling
from veles.znicz import normalization, dropout
from veles.znicz import activation
import veles.error as error


class TypeDict(UserDict):
    """
    All keys in the dictionary must be classes. Its [key] operator looks up
        the `key` inheritance hierarchy and chooses its nearest ancestor
        as a key to return its coupled value.
    """

    def __getitem__(self, key):
        if not isinstance(key, type):
            raise TypeError("key must be of class type")
        hierarchy = [key]
        while len(hierarchy):
            clazz = hierarchy.pop()
            val = self.data.get(clazz)
            if val is not None:
                return val
            elif clazz != type:
                hierarchy.extend(clazz.__bases__)
        raise KeyError("Unknown key %s" % str(key))


class GradientUnitFactory(object):
    """
    This factory makes :class:`GradientDescentBase`-interfaced units
        according to their forward-prop units.
    """
    _pooling_grad_classes = {pooling.AvgPooling: gd_pooling.GDAvgPooling,
                             pooling.MaxPooling: gd_pooling.GDMaxPooling}

    _conv_grad_classes = {conv.Conv: gd_conv.GradientDescentConv,
                          conv.ConvRELU: gd_conv.GDRELUConv,
                          conv.ConvStrictRELU: gd_conv.GDStrictRELUConv,
                          conv.ConvTanh: gd_conv.GDTanhConv}

    _all2all_grad_classes = {all2all.All2All: gd.GradientDescent,
                             all2all.All2AllRELU: gd.GDRELU,
                             all2all.All2AllTanh: gd.GDTanh,
                             all2all.All2AllSoftmax: gd.GDSM}

    _activation_grad_classes = {
        activation.ForwardTanh: activation.BackwardTanh,
        activation.ForwardRELU: activation.BackwardRELU,
        activation.ForwardStrictRELU: activation.BackwardStrictRELU,
        activation.ForwardLog: activation.BackwardLog,
        activation.ForwardTanhLog: activation.BackwardTanhLog,
        activation.ForwardSinCos: activation.BackwardSinCos,
        activation.ForwardMul: activation.BackwardMul
    }

    @staticmethod
    def create(fwd, name, **kwargs):
        """
        Creates gradient descent unit by forward prop unit.

        Args:
            fwd(:class:`Unit`): a forward propagation unit
            batch_size(int)
            learning_rate(float)
            bias_learning_rate(float): uses `learning_rate` if not set
            weight_decay(float)
            bias_weight_decay(float): uses `weight_decay` if not set
            momentum(float): 0 by default
            bias_momentum(float): uses `momentum` if not set

        Returns:
            :class:`GradientDescentBase`: a specific grad unit for `fwd_prop`
        """
        assert fwd is not None

        # Trick from  http://stackoverflow.com/a/3933062/2726900
        return GradientUnitFactory._methods_for_classes[type(fwd)].__get__(
            None, GradientUnitFactory)(fwd, name, **kwargs)

    @staticmethod
    def _create_grad_conv(fwd, name, **kwargs):
        grad_class = GradientUnitFactory._conv_grad_classes[type(fwd)]
        grad_unit = grad_class(
            fwd.workflow, name=name, kx=fwd.kx, ky=fwd.ky, sliding=fwd.sliding,
            padding=fwd.padding, n_kernels=fwd.n_kernels, **kwargs)
        grad_unit.link_attrs(fwd, "input", "output", "weights", "bias")
        return grad_unit

    @staticmethod
    def _create_grad_all2all(fwd, name, **kwargs):
        grad_class = GradientUnitFactory._all2all_grad_classes[type(fwd)]
        grad_unit = grad_class(fwd.workflow, name=name, **kwargs)
        grad_unit.link_attrs(fwd, "input", "output", "weights", "bias")
        return grad_unit

    @staticmethod
    def _create_grad_pooling(fwd, name, **kwargs):
        grad_class = GradientUnitFactory._pooling_grad_classes[type(fwd)]
        grad_unit = grad_class(
            fwd.workflow, name=name,
            kx=fwd.kx, ky=fwd.ky, sliding=fwd.sliding, **kwargs)
        grad_unit.link_attrs(fwd, "input", "output")
        if isinstance(fwd, pooling.MaxPooling):
            grad_unit.link_attrs(fwd, "input_offset")
        return grad_unit

    @staticmethod
    def _create_grad_activation(fwd, name, **kwargs):
        grad_class = GradientUnitFactory._activation_grad_classes[type(fwd)]
        grad_unit = grad_class(fwd.workflow, name=name, **kwargs)
        grad_unit.link_attrs(fwd, "input", "output")
        return grad_unit

    @staticmethod
    def _create_grad_lrn(fwd, name, **kwargs):
        grad_unit = normalization.LRNormalizerBackward(
            fwd.workflow, name=name, k=fwd.k, n=fwd.n,
            alpha=fwd.alpha, beta=fwd.beta, **kwargs)
        grad_unit.link_attrs(fwd, "input", "output")
        return grad_unit

    @staticmethod
    def _create_grad_dropout(fwd, name, **kwargs):
        grad_dropout = dropout.DropoutBackward(fwd.workflow, name=name)
        grad_dropout.link_attrs(fwd, "input", "output", "mask")
        return grad_dropout

    # calls this method for this BASE classes
    _methods_for_classes = TypeDict({
        conv.Conv: _create_grad_conv,
        pooling.Pooling: _create_grad_pooling,
        all2all.All2All: _create_grad_all2all,
        activation.ActivationForward: _create_grad_activation,
        normalization.LRNormalizerForward: _create_grad_lrn,
        dropout.DropoutForward: _create_grad_dropout
    })


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
            "stochastic_abs_pooling": (pooling.StochasticAbsPooling,
                                       gd_pooling.GDMaxAbsPooling),
            "stochastic_pooling": (pooling.StochasticPooling,
                                   gd_pooling.GDMaxPooling),
            "all2all": (all2all.All2All,
                        gd.GradientDescent),
            "all2all_relu": (all2all.All2AllRELU,
                             gd.GDRELU),
            "all2all_tanh": (all2all.All2AllTanh,
                             gd.GDTanh),
            "softmax": (all2all.All2AllSoftmax,
                        gd.GDSM),
            "activation_mul": (activation.ForwardMul,
                               activation.BackwardMul),
            "activation_tanh": (activation.ForwardTanh,
                                activation.BackwardTanh),
            "activation_tanhlog": (activation.ForwardTanhLog,
                                   activation.BackwardTanhLog),
            "activation_relu": (activation.ForwardRELU,
                                activation.BackwardRELU),
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
        if tpe not in self.layer_map:
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
        if len(self.fwds) > 0:
            prev_forward_unit = self.fwds[-1]
            new_unit.link_attrs(prev_forward_unit, ("input", "output"))
        else:
            assert self.loader is not None
            prev_forward_unit = self.loader
            new_unit.link_attrs(self.loader, ("input", "minibatch_data"))

        new_unit.link_from(prev_forward_unit)
        self.fwds.append(new_unit)

    def create_gd_units_by_config(self):
        """
        Creates GD units by config (`self.layers`)
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
                self.gds[i].link_from(self.gds[i + 1])
                self.gds[i].link_attrs(self.gds[i + 1],
                                       ("err_output", "err_input"))
            else:
                self.gds[-1].link_from(self.snapshotter)
                self.gds[-1].link_attrs(self.evaluator, "err_output")

            attrs = []
            for attr in ("input", "output", "weights", "bias",
                         "input_offset", "mask"):
                if hasattr(self.fwds[i], attr):
                    attrs.append(attr)
            self.gds[i].link_attrs(self.fwds[i], *attrs)

            self.gds[i].gate_skip = self.decision.gd_skip
            self.gds[i].link_attrs(self.loader,
                                   ("batch_size", "minibatch_size"))

        # Disable error backpropagation on the first layer
        self.gds[0].need_err_input = False

    def create_gd_units(self, lr_adjuster, **kwargs):
        """
        Creates gradient descent units for previously made self.fwds.
        Feeds their inputs with respect of their order.

        Args:
            lr_adjuster(:class:`LearningRateAdjust`): learning rate adjust unit
            lr_policy(:class:`veles.znicz.lr_adjust.ILRPolicy`)
                - should be set if **lr_adjuster is set**
            bias_lr_policy(:class:`veles.znicz.lr_adjust.ILRPolicy`)
                - should be set if **lr_adjuster is set**
            learning_rate
            learning_rate_bias
            weights_decay
            weights_decay_bias
            gradient_moment
            gradient_moment_bias
        """
        del self.gds[:]
        for fwd in self.fwds:
            self.gds.append(GradientUnitFactory.create(fwd, "gd_" + fwd.name,
                                                       **kwargs))
        for i, gd_elm in enumerate(self.gds[:-1]):
            gd_elm.link_from(self.gds[i + 1])
            gd_elm.link_attrs(self.gds[i + 1], ("err_output", "err_input"))

        if lr_adjuster is not None:
            lr_adjuster.link_from(self.snapshotter)
            for grad_unit in self.gds:
                lr_adjuster.add_gd_unit(grad_unit, kwargs["lr_policy"],
                                        kwargs["bias_lr_policy"])
            self.gds[-1].link_from(lr_adjuster)
        else:
            self.gds[-1].link_from(self.snapshotter)
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].gate_skip = self.decision.gd_skip

        for gd_elm in self.gds:
            gd_elm.link_attrs(self.loader, ("batch_size", "minibatch_size"))

        # Disable error backpropagation on the first layer
        self.gds[0].need_err_input = False
