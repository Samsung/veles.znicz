"""
Created on May 5, 2014

Standard workflow class definition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import six

if six.PY3:
    from collections import UserDict
else:
    from UserDict import UserDict

from veles.compat import from_none
import veles.error as error
from veles.interaction import Shell
from veles.mean_disp_normalizer import MeanDispNormalizer
import veles.plotting_units as plotting_units
# Important: do not remove unused imports! It will prevent MatchingObject
# metaclass from adding the mapping in the corresponding modules
from veles.znicz import nn_units
from veles.znicz import conv, pooling, all2all  # pylint: disable=W0611
from veles.znicz import gd, gd_conv, gd_pooling
from veles.znicz import normalization, dropout
from veles.znicz import activation
from veles.znicz.decision import DecisionGD, DecisionMSE
import veles.znicz.diversity as diversity
from veles.znicz.evaluator import EvaluatorSoftmax, EvaluatorMSE
import veles.znicz.image_saver as image_saver
from veles.loader.base import UserLoaderRegistry
import veles.znicz.lr_adjust as lr_adjust
import veles.znicz.nn_plotting_units as nn_plotting_units


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


class StandardWorkflowBase(nn_units.NNWorkflow):
    """
    A base class for standard workflows with forward and backward propagation.
    Is able to automatically create backward units by pre-created forward units
    """
    def __init__(self, workflow, **kwargs):
        super(StandardWorkflowBase, self).__init__(workflow, **kwargs)
        self.layer_map = nn_units.MatchingObject.mapping
        self.layers = kwargs.get("layers")
        self.device = kwargs.get("device")

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

    def parse_forwards_from_config(self, init_unit, init_attrs):
        """
        Parsing forward units from config.
        Adds a new fowrard unit to self.forwards, links it with previous
        fwd unit by link_from and link_attrs. If self.forwards is empty, links
        unit with self.loader
        """
        if type(self.layers) != list:
            raise error.BadFormatError("layers should be a list of dicts")
        del self.forwards[:]
        for i in range(len(self.layers)):
            layer = self.layers[i]
            tpe, kwargs = self._get_layer_type_kwargs(layer)
            try:
                unit = self.layer_map[tpe].forward(self, **kwargs)
            except IndexError:
                raise from_none(ValueError("Failed to find a Forward in %s" %
                                           tpe))
            self._add_forward_unit(unit, init_unit, init_attrs)

    def _add_forward_unit(self, new_unit, init_unit=None, init_attrs=None):
        """
        Adds a new fowrard unit to self.forwards, links it with previous fwd
        unit by link_from and link_attrs. If self.forwards is empty, links unit
        with self.loader
        """
        if len(self.forwards) > 0:
            prev_forward_unit = self.forwards[-1]
            new_unit.link_attrs(prev_forward_unit, ("input", "output"))
        else:
            if init_unit is None:
                raise ValueError("init_unit is None for first fwd!")
            prev_forward_unit = init_unit
            new_unit.link_attrs(init_unit, init_attrs)

        new_unit.link_from(prev_forward_unit)
        self.forwards.append(new_unit)

    def create_gd_units_by_config(self, init_unit):
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
            if not isinstance(self.forwards[i], self.layer_map[tpe].forward):
                raise error.BadFormatError(
                    "Forward layer %s at position %d "
                    "is not an instance of %s" %
                    (self.forwards[i], i, self.layer_map[tpe].forward))

            if "name" in kwargs:
                kwargs["name"] = "gd_" + kwargs["name"]
            unit = next(self.layer_map[tpe].backwards)(self, **kwargs)
            self.gds[i] = unit

            # Link attributes
            if i < len(self.forwards) - 1:
                self.gds[i].link_from(self.gds[i + 1])
                self.gds[i].link_attrs(self.gds[i + 1],
                                       ("err_output", "err_input"))
            else:
                self.gds[-1].link_from(init_unit)
                self.gds[-1].link_attrs(self.evaluator, "err_output")

            attrs = []
            for attr in ("input", "output", "weights", "bias",
                         "input_offset", "mask"):
                if hasattr(self.forwards[i], attr):
                    attrs.append(attr)
            self.gds[i].link_attrs(self.forwards[i], *attrs)

            self.gds[i].gate_skip = self.decision.gd_skip
            self.gds[i].link_attrs(self.loader,
                                   ("batch_size", "minibatch_size"))

        # Disable error backpropagation on the first layer
        self.gds[0].need_err_input = False

    def create_gd_units(self, lr_adjuster, **kwargs):
        """
        Creates gradient descent units for previously made self.forwards.
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
        for fwd in self.forwards:
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


class StandardWorkflow(StandardWorkflowBase):
    """
    Workflow for trivially connections between Unit.
    User can create Self-constructing Models with that class.
    It means that User can change structure of Model (Convolutional,
    Fully connected, different parameters) and parameters of training in
    configuration file.
    attributes:
        loss_function: name of Loss function. Choices are "softmax" or "mse"
        loader_name: name of Loader. If loader_name is None, User should
            redefine link_loader() function and create own Loader
        loader_config: loader configuration parameters
        decision_config: decision configuration parameters
        snapshotter_config: snapshotter configuration parameters
        image_saver_config: image_saver configuration parameters
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers", [{}])
        kwargs["layers"] = layers
        super(StandardWorkflow, self).__init__(
            workflow, **kwargs)
        self.loader_config = kwargs.get("loader_config")
        self.decision_config = kwargs.get("decision_config")
        self.snapshotter_config = kwargs.get("snapshotter_config")
        self.loss_function = kwargs.get("loss_function", "softmax")
        self.image_saver_config = kwargs.get("image_saver_config")
        self.loader_name = kwargs.get("loader_name")
        if self.loss_function != "softmax" and self.loss_function != "mse":
            raise error.NotExistsError("Unknown loss function type %s"
                                       % self.loss_function)

        self.create_workflow()

    def create_workflow(self):
        self.link_repeater(self.start_point)

        self.link_loader(self.repeater)

        # Add forwards units
        self.link_forwards(self.loader, ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.link_evaluator(self.forwards[-1])

        # Add decision unit
        self.link_decision(self.evaluator)

        # Add snapshotter unit
        self.link_snapshotter(self.decision)

        # Add gradient descent units
        self.link_gds(self.snapshotter)

        self.link_end_point(self.gds[0])

    def link_repeater(self, init_unit):
        self.repeater.link_from(init_unit)

    def link_loader(self, init_unit):
        if self.loader_name is None:
            raise AttributeError(
                "Set the loader_name. Full list of names is %s. Or redefine"
                " link_loader() function, if you want to create you own Loader"
                % list(UserLoaderRegistry.loaders.keys()))
        self.loader = UserLoaderRegistry.loaders[self.loader_name](
            self, **self.loader_config.__dict__)
        self.loader.link_from(init_unit)
        pass

    def link_forwards(self, init_unit, init_attrs):
        self.parse_forwards_from_config(init_unit, init_attrs)

    def check_decision(self):
        if self.decision is None:
            raise error.NotExistsError(
                "Please create decision in workflow first."
                "For that you can use link_decision() function")

    def check_evaluator(self):
        if self.evaluator is None:
            raise error.NotExistsError(
                "Please create evaluator in workflow first."
                "For that you can use link_evaluator() function")

    def check_forwards(self):
        if self.forwards is None:
            raise error.NotExistsError(
                "Please create forwards in workflow first."
                "You can use link_forwards() function")

    def check_loader(self):
        if self.loader is None:
            raise error.NotExistsError(
                "Please create loader in workflow first."
                "For that you can use link_loader() function")

    def check_gds(self):
        if self.gds is None:
            raise error.NotExistsError(
                "Please create gds in workflow first."
                "For that you can use link_gds() function")

    def link_gds(self, init_unit):
        # not work without create forwards, evaluator and decision first
        self.check_evaluator()
        self.check_forwards()
        self.check_decision()
        self.create_gd_units_by_config(init_unit)
        self.gds[-1].unlink_before()
        self.gds[-1].link_from(init_unit)

    def link_evaluator(self, init_unit):
        # not work without create forwards and loader first
        self.check_forwards()
        self.check_loader()
        self.evaluator = (
            EvaluatorSoftmax(self) if self.loss_function == "softmax"
            else EvaluatorMSE(self))
        self.evaluator.link_from(init_unit)
        self.evaluator.link_attrs(self.forwards[-1], "output")
        if self.loss_function == "softmax":
            self.evaluator.link_attrs(self.forwards[-1], "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("labels", "minibatch_labels"),
                                  ("max_samples_per_epoch", "total_samples"))
        if self.loss_function == "mse":
            self.evaluator.link_attrs(self.loader,
                                      ("target", "minibatch_targets"),
                                      "class_targets")

    def link_decision(self, init_unit):
        # not work without create loader and evaluator first
        self.check_loader()
        self.check_evaluator()
        self.decision = (DecisionGD(self, **self.decision_config.__dict__)
                         if self.loss_function == "softmax" else DecisionMSE(
                             self, **self.decision_config.__dict__))
        self.decision.link_from(init_unit)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "last_minibatch",
                                 "minibatch_size",
                                 "class_lengths",
                                 "epoch_ended",
                                 "epoch_number")
        if self.loss_function == "mse":
            self.decision.link_attrs(self.loader, "minibatch_offset")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"))
        if self.loss_function == "softmax":
            self.decision.link_attrs(
                self.evaluator,
                ("minibatch_confusion_matrix", "confusion_matrix"),
                ("minibatch_max_err_y_sum", "max_err_output_sum"))
        if self.loss_function == "mse":
            self.decision.link_attrs(
                self.evaluator,
                ("minibatch_metrics", "metrics"),
                ("minibatch_mse", "mse"))

    def link_snapshotter(self, init_unit):
        # not work without create decision first
        self.check_decision()
        self.snapshotter = nn_units.NNSnapshotter(
            self, **self.snapshotter_config.__dict__)
        self.snapshotter.link_from(init_unit)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.decision.epoch_ended | ~self.decision.improved)

    def link_image_saver(self, init_unit):
        # not work without create forwards, loader and decision first
        self.check_forwards()
        self.check_decision()
        self.check_loader()
        self.image_saver = image_saver.ImageSaver(
            self, **self.image_saver_config.__dict__)
        self.image_saver.link_from(init_unit)
        self.image_saver.link_attrs(self.forwards[-1],
                                    "output", "max_idx")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indexes", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")

    def link_lr_adjuster(self, init_unit):
        # not work without create gds first
        self.check_gds()
        self.lr_adjuster = lr_adjust.LearningRateAdjust(self)
        for gd_elm in self.gds:
            self.lr_adjuster.add_gd_unit(
                gd_elm,
                lr_policy=lr_adjust.ArbitraryStepPolicy(
                    [(gd_elm.learning_rate, 60000),
                     (gd_elm.learning_rate / 10., 5000),
                     (gd_elm.learning_rate / 100., 100000000)]),
                bias_lr_policy=lr_adjust.ArbitraryStepPolicy(
                    [(gd_elm.learning_rate_bias, 60000),
                     (gd_elm.learning_rate_bias / 10., 5000),
                     (gd_elm.learning_rate_bias / 100., 100000000)])
                )
        self.lr_adjuster.link_from(init_unit)

    def link_meandispnorm(self, init_unit):
        # not work without create loader first
        self.check_loader()
        self.meandispnorm = MeanDispNormalizer(self)
        self.meandispnorm.link_attrs(self.loader,
                                     ("input", "minibatch_data"),
                                     "mean", "rdisp")
        self.meandispnorm.link_from(init_unit)

    def link_ipython(self, init_unit):
        # not work without create decision first
        self.check_decision()
        self.ipython = Shell(self)
        self.ipython.link_from(init_unit)
        self.ipython.gate_skip = ~self.decision.epoch_ended

    def link_error_plotter(self, init_unit):
        # not work without create decision first
        self.check_decision()
        self.error_plotter = []
        prev = init_unit
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.error_plotter.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.error_plotter[-1].link_attrs(self.decision,
                                              ("input", "epoch_n_err_pt"))
            self.error_plotter[-1].input_field = i
            self.error_plotter[-1].link_from(prev)
            self.error_plotter[-1].gate_skip = ~self.decision.epoch_ended
            prev = self.error_plotter[-1]
        self.error_plotter[0].clear_plot = True
        self.error_plotter[-1].redraw_plot = True

    def link_conf_matrix_plotter(self, init_unit):
        # not work without create decision first
        self.check_decision()
        self.conf_matrix_plotter = []
        prev = init_unit
        for i in range(1, len(self.decision.confusion_matrixes)):
            self.conf_matrix_plotter.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.conf_matrix_plotter[-1].link_attrs(
                self.decision, ("input", "confusion_matrixes"))
            self.conf_matrix_plotter[-1].input_field = i
            self.conf_matrix_plotter[-1].link_from(prev)
            self.conf_matrix_plotter[-1].gate_skip = ~self.decision.epoch_ended
            prev = self.conf_matrix_plotter[-1]

    def link_err_y_plotter(self, init_unit):
        # not work without create decision first
        self.check_decision()
        styles = ["r-", "b-", "k-"]
        self.err_y_plotter = []
        prev = init_unit
        for i in range(1, 3):
            self.err_y_plotter.append(plotting_units.AccumulatingPlotter(
                self, name="Last layer max gradient sum",
                plot_style=styles[i]))
            self.err_y_plotter[-1].link_attrs(
                self.decision, ("input", "max_err_y_sums"))
            self.err_y_plotter[-1].input_field = i
            self.err_y_plotter[-1].link_from(prev)
            self.err_y_plotter[-1].gate_skip = ~self.decision.epoch_ended
            prev = self.err_y_plotter[-1]
        self.err_y_plotter[0].clear_plot = True
        self.err_y_plotter[-1].redraw_plot = True

    def link_multi_hist_plotter(self, init_unit, layers):
        # not work without create decision, forwards first and set layers
        self.check_decision()
        self.check_forwards()
        self.multi_hist_plotter = []
        prev = init_unit
        for i in range(0, len(layers)):
            multi_hist = plotting_units.MultiHistogram(
                self, name="Histogram %s %s" % (i + 1, layers[i]["type"]))
            self.multi_hist_plotter.append(multi_hist)
            if layers[i].get("n_kernels") is not None:
                self.multi_hist_plotter[-1].link_from(prev)
                prev = self.multi_hist_plotter[-1]
                self.multi_hist_plotter[
                    - 1].hist_number = layers[i]["n_kernels"]
                self.multi_hist_plotter[-1].link_attrs(
                    self.forwards[i], ("input", "weights"))
                end_epoch = ~self.decision.epoch_ended
                self.multi_hist_plotter[-1].gate_skip = end_epoch
            if layers[i].get("output_shape") is not None:
                self.multi_hist_plotter[-1].link_from(prev)
                prev = self.multi_hist_plotter[-1]
                self.multi_hist_plotter[
                    - 1].hist_number = layers[i]["output_shape"]
                self.multi_hist_plotter[-1].link_attrs(
                    self.forwards[i], ("input", "weights"))
                self.multi_hist_plotter[
                    - 1].gate_skip = ~self.decision.epoch_ended

    def link_weights_plotter(self, init_unit, layers, limit, weights_input):
        # not work without create decision, forwards first and set layers,
        # limit and weights_input - "weights" or "gradient_weights" for example
        self.check_decision()
        self.check_forwards()
        prev = init_unit
        self.weights_plotter = []
        prev_channels = 3
        for i in range(0, len(layers)):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All)):
                continue
            plt_wd = nn_plotting_units.Weights2D(
                self, name="%s %s" % (i + 1, layers[i]["type"]),
                limit=limit)
            self.weights_plotter.append(plt_wd)
            self.weights_plotter[-1].link_attrs(self.forwards[i],
                                                ("input", weights_input))
            self.weights_plotter[-1].input_field = "mem"
            if isinstance(self.forwards[i], conv.Conv):
                self.weights_plotter[-1].get_shape_from = (
                    [self.forwards[i].kx, self.forwards[i].ky, prev_channels])
                prev_channels = self.forwards[i].n_kernels
            if (layers[i].get("output_shape") is not None and
                    layers[i]["type"] != "softmax"):
                self.weights_plotter[-1].link_attrs(
                    self.forwards[i], ("get_shape_from", "input"))
            self.weights_plotter[-1].link_from(prev)
            prev = self.weights_plotter[-1]
            self.weights_plotter[-1].gate_skip = ~self.decision.epoch_ended

    def link_similar_weights_plotter(self, init_unit, layers, limit,
                                     magnitude, form, peak):
        # not work without create weights_plotter, decision and forwards first
        self.check_decision()
        self.check_forwards()
        self.similar_weights_plotter = []
        prev = init_unit
        k = 0
        n = 0
        for i in range(len(layers)):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All)):
                k += 1
                n = i - k
                continue
            plt_mx = diversity.SimilarWeights2D(
                self, name="%s %s [similar]" % (i + 1, layers[i]["type"]),
                limit=limit,
                form_threshold=form,
                peak_threshold=peak,
                magnitude_threshold=magnitude)
            self.similar_weights_plotter.append(plt_mx)
            self.similar_weights_plotter[-1].link_attrs(self.forwards[i],
                                                        ("input", "weights"))
            self.similar_weights_plotter[-1].input_field = "mem"
            wd_plt = self.weights_plotter
            if n != 0:
                self.similar_weights_plotter[
                    - 1].get_shape_from = wd_plt[n].get_shape_from
            if (layers[i].get("output_shape") is not None and
                    layers[i]["type"] != "softmax"):
                self.similar_weights_plotter[-1].link_attrs(
                    self.forwards[i], ("get_shape_from", "input"))
            self.similar_weights_plotter[-1].link_from(prev)
            prev = self.similar_weights_plotter[-1]
            self.similar_weights_plotter[
                - 1].gate_skip = ~self.decision.epoch_ended
        self.similar_weights_plotter[0].clear_plot = True
        self.similar_weights_plotter[-1].redraw_plot = True

    def link_table_plotter(self, layers, init_unit):
        # not work without create decision and forwards first
        self.check_decision()
        self.check_forwards()
        self.table_plotter = plotting_units.TableMaxMin(self, name="Max, Min")
        del self.table_plotter.y[:]
        del self.table_plotter.col_labels[:]
        for i in range(0, len(layers)):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All)):
                continue
            obj = self.forwards[i].weights
            name = "weights %s %s" % (i + 1, layers[i]["type"])
            self.table_plotter.y.append(obj)
            self.table_plotter.col_labels.append(name)
            obj = self.gds[i].gradient_weights
            name = "gd %s %s" % (i + 1, layers[i]["type"])
            self.table_plotter.y.append(obj)
            self.table_plotter.col_labels.append(name)
            obj = self.forwards[i].output
            name = "Y %s %s" % (i + 1, layers[i]["type"])
            self.table_plotter.y.append(obj)
            self.table_plotter.col_labels.append(name)
        self.table_plotter.link_from(init_unit)
        self.table_plotter.gate_skip = ~self.decision.epoch_ended

    def link_mse_plotter(self, init_unit):
        # not work without create decision first
        self.check_decision()
        prev = init_unit
        self.mse_plotter = []
        styles = ["", "", "k-"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.mse_plotter.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.mse_plotter[-1].link_attrs(
                self.decision, ("input", "epoch_metrics"))
            self.mse_plotter[-1].input_field = i
            self.mse_plotter[-1].link_from(prev)
            prev = self.mse_plotter[-1]
            self.mse_plotter[-1].gate_skip = ~self.decision.epoch_ended
        self.mse_plotter[0].clear_plot = True

    def link_max_plotter(self, init_unit):
        # not work without create decision first
        self.check_decision()
        prev = init_unit
        self.max_plotter = []
        styles = ["", "", "k--"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.max_plotter.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.max_plotter[-1].link_attrs(
                self.decision, ("input", "epoch_metrics"))
            self.max_plotter[-1].input_field = i
            self.max_plotter[-1].input_offset = 1
            self.max_plotter[-1].link_from(prev)
            prev = self.max_plotter[-1]
            self.max_plotter[-1].gate_skip = ~self.decision.epoch_ended

    def link_min_plotter(self, init_unit):
        # not work without create decision first
        self.check_decision()
        prev = init_unit
        self.min_plotter = []
        styles = ["", "", "k:"]  # ["r:", "b:", "k:"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.min_plotter.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.min_plotter[-1].link_attrs(
                self.decision, ("input", "epoch_metrics"))
            self.min_plotter[-1].input_field = i
            self.min_plotter[-1].input_offset = 2
            self.min_plotter[-1].link_from(prev)
            prev = self.min_plotter[-1]
            self.min_plotter[-1].gate_skip = ~self.decision.epoch_ended
        self.min_plotter[-1].redraw_plot = True

    def link_image_plotter(self, init_unit):
        # not work without create decision and forwards first
        self.check_decision()
        self.check_forwards()
        self.image_plotter = plotting_units.ImagePlotter(self,
                                                         name="output sample")
        self.image_plotter.inputs.append(self.forwards[-1].output)
        self.image_plotter.input_fields.append(0)
        self.image_plotter.inputs.append(self.forwards[0].input)
        self.image_plotter.input_fields.append(0)
        self.image_plotter.link_from(init_unit)
        self.image_plotter.gate_skip = ~self.decision.epoch_ended

    def link_immediate_plotter(self, init_unit):
        # not work without create decision, loader and forwards first
        self.check_decision()
        self.check_forwards()
        self.check_loader()
        self.immediate_plotter = plotting_units.ImmediatePlotter(
            self, name="ImmediatePlotter", ylim=[-1.1, 1.1])
        del self.immediate_plotter.inputs[:]
        self.immediate_plotter.inputs.append(self.loader.minibatch_data)
        self.immediate_plotter.inputs.append(self.loader.minibatch_targets)
        self.immediate_plotter.inputs.append(self.forwards[-1].output)
        del self.immediate_plotter.input_fields[:]
        self.immediate_plotter.input_fields.append(0)
        self.immediate_plotter.input_fields.append(0)
        self.immediate_plotter.input_fields.append(0)
        del self.immediate_plotter.input_styles[:]
        self.immediate_plotter.input_styles.append("k-")
        self.immediate_plotter.input_styles.append("g-")
        self.immediate_plotter.input_styles.append("b-")
        self.immediate_plotter.link_from(init_unit)
        self.immediate_plotter.gate_skip = ~self.decision.epoch_ended

    def link_end_point(self, init_unit):
        # not work without create decision first
        self.check_decision()
        self.repeater.link_from(init_unit)
        self.end_point.link_from(init_unit)
        self.end_point.gate_block = ~self.decision.complete
        self.loader.gate_block = self.decision.complete
