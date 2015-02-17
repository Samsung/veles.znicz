"""
Created on May 5, 2014

Standard workflow class definition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import six
from veles.znicz.conv import ConvolutionalBase
from veles.znicz.gd_pooling import GDPooling

if six.PY3:
    from collections import UserDict
else:
    from UserDict import UserDict

from veles.compat import from_none
import veles.error as error
from veles.interaction import Shell
from veles.loader.saver import MinibatchesSaver
from veles.mean_disp_normalizer import MeanDispNormalizer
import veles.plotting_units as plotting_units
# Important: do not remove unused imports! It will prevent MatchingObject
# metaclass from adding the mapping in the corresponding modules
from veles.znicz import nn_units
from veles.znicz import conv, pooling, all2all,\
    weights_zerofilling  # pylint: disable=W0611
from veles.znicz import gd, gd_conv, gd_pooling
from veles.znicz import normalization, dropout
from veles.znicz import activation
from veles.znicz.decision import DecisionGD, DecisionMSE
import veles.znicz.diversity as diversity
from veles.znicz.evaluator import EvaluatorSoftmax, EvaluatorMSE
import veles.znicz.image_saver as image_saver
from veles.loader.base import UserLoaderRegistry
from veles.loader.image import ImageLoader
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
                             all2all.All2AllSoftmax: gd.GDSoftmax}

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
        grad_unit = grad_class(fwd.workflow, name=name, **kwargs)
        grad_unit.link_attrs(fwd, "input", "output", "weights", "bias")
        grad_unit.link_conv_attrs(fwd)
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
        grad_unit = grad_class(fwd.workflow, name=name, **kwargs)
        grad_unit.link_attrs(fwd, "input", "output")
        if isinstance(fwd, pooling.MaxPooling):
            grad_unit.link_attrs(fwd, "input_offset")
        grad_unit.link_pool_attrs(fwd)
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
        self.layers = kwargs.get("layers", [{}])
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
        kwargs_forward = dict(layer.get("->", {}))
        kwargs_backward = dict(layer.get("<-", {}))
        # Add shared parameters to both dicts
        others = {k: v for k, v in layer.items()
                  if k not in ("type", "->", "<-", "name")}
        kwargs_forward.update(others)
        kwargs_backward.update(others)
        if "name" in layer:
            kwargs_forward["name"] = layer["name"] + "_forward"
            kwargs_backward["name"] = layer["name"] + "_backward"
        return tpe, kwargs_forward, kwargs_backward

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
        for i, layer in enumerate(self.layers):
            tpe, kwargs, _ = self._get_layer_type_kwargs(layer)
            try:
                unit = self.layer_map[tpe].forward(self, **kwargs)
            except IndexError:
                raise from_none(ValueError("Failed to find a Forward in %s" %
                                           tpe))
            self._add_forward_unit(unit, init_unit, init_attrs)
        # Another loop for ZeroFiller unit. Linking attributes for
        # ZeroFiller from attributes of next layer
        for prev_forward, forward in zip(self.forwards, self.forwards[1:]):
            if isinstance(prev_forward, weights_zerofilling.ZeroFiller):
                prev_forward.link_attrs(forward, "weights")
            prev_forward = forward

    def _add_forward_unit(self, new_unit, init_unit=None, init_attrs=None):
        """
        Adds a new fowrard unit to self.forwards, links it with previous fwd
        unit by link_from and link_attrs. If self.forwards is empty, links unit
        with self.loader
        """
        if len(self.forwards) > 0:
            prev_forward_unit = self.forwards[-1]
        else:
            if init_unit is None:
                raise ValueError("init_unit is None for first fwd!")
            prev_forward_unit = init_unit

        new_unit.link_from(prev_forward_unit)

        fwds_with_attrs = tuple(
            filter(
                lambda fwd: not isinstance(
                    fwd, weights_zerofilling.ZeroFiller), self.forwards))

        self.forwards.append(new_unit)

        if isinstance(new_unit, weights_zerofilling.ZeroFiller):
            return

        if len(fwds_with_attrs) > 0:
            new_unit.link_attrs(fwds_with_attrs[-1], ("input", "output"))
        else:
            new_unit.link_attrs(init_unit, init_attrs)
        del fwds_with_attrs

    def create_gd_units_by_config(self, init_unit):
        """
        Creates GD units by config (`self.layers`)
        """
        if type(self.layers) != list:
            raise error.BadFormatError("layers should be a list of dicts")
        del self.gds[:]
        self.gds.extend(None for _ in self.layers)
        last_gd = None
        units_to_delete = []
        for i, layer in reversed(list(enumerate(self.layers))):
            tpe, _, kwargs = self._get_layer_type_kwargs(layer)

            # Check corresponding forward unit type
            if not isinstance(self.forwards[i], self.layer_map[tpe].forward):
                raise error.BadFormatError(
                    "Forward layer %s at position %d "
                    "is not an instance of %s" %
                    (self.forwards[i], i, self.layer_map[tpe].forward))

            if "name" in kwargs:
                kwargs["name"] = "gd_" + kwargs["name"]
            try:
                unit = next(self.layer_map[tpe].backwards)(self, **kwargs)
            except StopIteration:
                units_to_delete.append(i)
                continue

            self.gds[i] = unit

            # Link attributes
            if last_gd is not None:
                unit.link_from(last_gd)
                unit.link_attrs(last_gd, ("err_output", "err_input"))
            else:
                unit.link_from(init_unit)
                unit.link_attrs(self.evaluator, "err_output")

            attrs = []
            # TODO(v.markovtsev): add "wants" to Unit and use it here
            try_link_attrs = set(["input", "weights", "bias", "input_offset",
                                  "mask", "output"])
            if isinstance(unit, ConvolutionalBase):
                try_link_attrs.update(ConvolutionalBase.CONV_ATTRS)
            if isinstance(unit, GDPooling):
                try_link_attrs.update(GDPooling.POOL_ATTRS)
            for attr in try_link_attrs:
                if hasattr(self.forwards[i], attr):
                    attrs.append(attr)
            unit.link_attrs(self.forwards[i], *attrs)

            unit.gate_skip = self.decision.gd_skip
            unit.link_attrs(self.loader, ("batch_size", "minibatch_size"))

            last_gd = unit

        # Remove None elements
        for i in units_to_delete:
            del self.gds[i]

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
        self.loader_config = kwargs.pop("loader_config")
        self.decision_config = kwargs.pop("decision_config", None)
        self.snapshotter_config = kwargs.pop("snapshotter_config", None)
        self.loss_function = kwargs.pop("loss_function", "softmax")
        self.image_saver_config = kwargs.pop("image_saver_config", None)
        self.data_saver_config = kwargs.pop("data_saver_config", None)
        self.similar_weights_plotter_config = kwargs.pop(
            "similar_weights_plotter_config", None)
        super(StandardWorkflow, self).__init__(workflow, **kwargs)
        self.loader_name = kwargs["loader_name"]
        if self.loss_function not in ("softmax", "mse"):
            raise ValueError(
                "Unknown loss function type %s" % self.loss_function)

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
        if self.loader_name not in list(UserLoaderRegistry.loaders.keys()):
            raise AttributeError(
                "Set the loader_name. Full list of names is %s. Or redefine"
                " link_loader() function, if you want to create you own Loader"
                % list(UserLoaderRegistry.loaders.keys()))
        self.loader = UserLoaderRegistry.loaders[self.loader_name](
            self, **self.loader_config.__content__)
        self.loader.link_from(init_unit)
        pass

    def link_data_saver(self, init_unit):
        if self.data_saver_config is not None:
            kwargs = self.data_saver_config.__content__
        else:
            kwargs = {}
        self.data_saver = MinibatchesSaver(
            self, **kwargs)
        self.data_saver.link_from(init_unit)
        self.data_saver.link_attrs(
            self.loader, "shuffle_limit", "minibatch_class", "minibatch_data",
            "minibatch_labels", "class_lengths", "max_minibatch_size",
            "minibatch_size")

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
        if self.decision_config is not None:
            kwargs = self.decision_config.__content__
        else:
            kwargs = {}
        self.decision = (DecisionGD(self, **kwargs)
                         if self.loss_function == "softmax" else DecisionMSE(
                             self, **kwargs))
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
        if self.snapshotter_config is not None:
            kwargs = self.snapshotter_config.__content__
        else:
            kwargs = {}
        self.snapshotter = nn_units.NNSnapshotter(
            self, **kwargs)
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
        if self.image_saver_config is not None:
            kwargs = self.image_saver_config.__content__
        else:
            kwargs = {}
        self.image_saver = image_saver.ImageSaver(
            self, **kwargs)
        self.image_saver.link_from(init_unit)
        if self.loss_function == "softmax":
            self.image_saver.link_attrs(self.forwards[-1], "max_idx")
        self.image_saver.link_attrs(self.forwards[-1], "output")
        if isinstance(self.loader, ImageLoader):
            self.image_saver.link_attrs(self.loader, "color_space")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indices", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")
        if self.loss_function == "mse":
            self.image_saver.link_attrs(
                self.loader, ("target", "minibatch_targets"))
        self.image_saver.gate_skip = ~self.decision.improved
        self.image_saver.link_attrs(self.snapshotter,
                                    ("this_save_time", "time"))

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
        for i in 1, 2:
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
        for i in 1, 2:
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

    def link_multi_hist_plotter(self, init_unit, layers, weights_input):
        # not work without create decision, forwards first and set layers
        self.check_decision()
        self.check_forwards()
        self.multi_hist_plotter = []
        prev = init_unit
        if weights_input == "weights":
            link_units = self.forwards
        elif weights_input == "gradient_weights":
            link_units = self.gds
        else:
            raise AttributeError(
                "weights_input should be 'weights' or 'gradient_weights'")
        for i, layer in enumerate(layers):
            multi_hist = plotting_units.MultiHistogram(
                self, name="Histogram %s %s" % (i + 1, layer["type"]))
            self.multi_hist_plotter.append(multi_hist)
            if layer.get("n_kernels") is not None:
                self.multi_hist_plotter[-1].link_from(prev)
                prev = self.multi_hist_plotter[-1]
                self.multi_hist_plotter[
                    - 1].hist_number = layer["n_kernels"]
                self.multi_hist_plotter[-1].link_attrs(
                    link_units[i], ("input", weights_input))
                end_epoch = ~self.decision.epoch_ended
                self.multi_hist_plotter[-1].gate_skip = end_epoch
            if layer.get("output_sample_shape") is not None:
                self.multi_hist_plotter[-1].link_from(prev)
                prev = self.multi_hist_plotter[-1]
                self.multi_hist_plotter[
                    - 1].hist_number = layer["output_sample_shape"]
                self.multi_hist_plotter[-1].link_attrs(
                    link_units[i], ("input", weights_input))
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
        if weights_input == "weights":
            link_units = self.forwards
        elif weights_input == "gradient_weights":
            link_units = self.gds
        else:
            raise AttributeError(
                "weights_input should be 'weights' or 'gradient_weights'")
        for i, layer in enumerate(layers):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All)):
                continue
            plt_wd = nn_plotting_units.Weights2D(
                self, name="%s %s" % (i + 1, layer["type"]),
                limit=limit)
            self.weights_plotter.append(plt_wd)
            self.weights_plotter[-1].link_attrs(link_units[i],
                                                ("input", weights_input))
            if isinstance(self.loader, ImageLoader):
                self.weights_plotter[-1].link_attrs(self.loader, "color_space")
            self.weights_plotter[-1].input_field = "mem"
            if isinstance(self.forwards[i], conv.Conv):
                self.weights_plotter[-1].get_shape_from = (
                    [self.forwards[i].kx, self.forwards[i].ky, prev_channels])
                prev_channels = self.forwards[i].n_kernels
            if (layer.get("output_sample_shape") is not None and
                    layer["type"] != "softmax"):
                self.weights_plotter[-1].link_attrs(
                    self.forwards[i], ("get_shape_from", "input"))
                if isinstance(self.loader, ImageLoader):
                    self.weights_plotter[-1].link_attrs(
                        self.loader, "color_space")
            self.weights_plotter[-1].link_from(prev)
            prev = self.weights_plotter[-1]
            self.weights_plotter[-1].gate_skip = ~self.decision.epoch_ended

    def link_similar_weights_plotter(self, init_unit, layers, weights_input):
        # not work without create weights_plotter, decision and forwards first
        self.check_decision()
        self.check_forwards()
        self.similar_weights_plotter = []
        prev = init_unit
        k = 0
        n = 0
        if weights_input == "weights":
            link_units = self.forwards
        elif weights_input == "gradient_weights":
            link_units = self.gds
        else:
            raise AttributeError(
                "weights_input should be 'weights' or 'gradient_weights'")
        for i, layer in enumerate(layers):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All)):
                k += 1
                n = i - k
                continue
            plt_mx = diversity.SimilarWeights2D(
                self, name="%s %s [similar]" % (i + 1, layer["type"]),
                **self.similar_weights_plotter_config.__content__)
            self.similar_weights_plotter.append(plt_mx)
            self.similar_weights_plotter[-1].link_attrs(
                link_units[i], ("input", weights_input))
            if isinstance(self.loader, ImageLoader):
                self.similar_weights_plotter[-1].link_attrs(
                    self.loader, "color_space")
            self.similar_weights_plotter[-1].input_field = "mem"
            wd_plt = self.weights_plotter
            if n != 0:
                self.similar_weights_plotter[
                    - 1].get_shape_from = wd_plt[n].get_shape_from
            if (layer.get("output_sample_shape") is not None and
                    layer["type"] != "softmax"):
                self.similar_weights_plotter[-1].link_attrs(
                    self.forwards[i], ("get_shape_from", "input"))
                if isinstance(self.loader, ImageLoader):
                    self.similar_weights_plotter[-1].link_attrs(
                        self.loader, "color_space")
            self.similar_weights_plotter[-1].link_from(prev)
            prev = self.similar_weights_plotter[-1]
            self.similar_weights_plotter[
                - 1].gate_skip = ~self.decision.epoch_ended
        self.similar_weights_plotter[0].clear_plot = True
        self.similar_weights_plotter[-1].redraw_plot = True

    def link_table_plotter(self, init_unit, layers):
        # not work without create decision and forwards first
        self.check_decision()
        self.check_forwards()
        self.table_plotter = plotting_units.TableMaxMin(self, name="Max, Min")
        del self.table_plotter.y[:]
        del self.table_plotter.col_labels[:]
        for i, layer in enumerate(layers):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All)):
                continue
            obj = self.forwards[i].weights
            name = "weights %s %s" % (i + 1, layer["type"])
            self.table_plotter.y.append(obj)
            self.table_plotter.col_labels.append(name)
            obj = self.gds[i].gradient_weights
            name = "gd %s %s" % (i + 1, layer["type"])
            self.table_plotter.y.append(obj)
            self.table_plotter.col_labels.append(name)
            obj = self.forwards[i].output
            name = "Y %s %s" % (i + 1, layer["type"])
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
        for i, style in enumerate(styles):
            if len(style) == 0:
                continue
            self.mse_plotter.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=style))
            self.mse_plotter[-1].link_attrs(
                self.decision, ("input", "epoch_metrics"))
            self.mse_plotter[-1].input_field = i
            self.mse_plotter[-1].link_from(prev)
            prev = self.mse_plotter[-1]
            self.mse_plotter[-1].gate_skip = ~self.decision.epoch_ended
        self.mse_plotter[0].clear_plot = True

    def link_min_max_plotter(self, init_unit, is_min):
        """
        :param is_min: True if linking min plotter, otherwise, False for max.
        :return: None.
        """
        # Requires Decision unit to be linked first.
        self.check_decision()
        prev = init_unit
        if is_min:
            plotter = self.min_plotter = []
        else:
            plotter = self.max_plotter = []
        styles = ["", "", "k:" if is_min else "k--"]
        for i, style in enumerate(styles):
            if len(style) == 0:
                continue
            plotter.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=style))
            plotter[-1].link_attrs(
                self.decision, ("input", "epoch_metrics"))
            plotter[-1].input_field = i
            plotter[-1].input_offset = 2 if is_min else 1
            plotter[-1].link_from(prev)
            prev = plotter[-1]
            plotter[-1].gate_skip = ~self.decision.epoch_ended
        plotter[-1].redraw_plot = True

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
