# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 5, 2014

Standard workflow class definition.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


from collections import namedtuple
import numpy
import re
import six
from zope.interface import implementer
from veles.avatar import Avatar
from veles.distributable import IDistributable, TriviallyDistributable
from veles.pickle2 import best_protocol
from veles.plumbing import FireStarter
from veles.snapshotter import SnapshotterRegistry
from veles.units import Unit, IUnit
from veles.znicz.diff_stats import DiffStats
from veles.publisher import Publisher

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
from veles.znicz import conv, pooling, all2all, weights_zerofilling
from veles.znicz import gd, gd_conv, gd_pooling
from veles.znicz import normalization, dropout
from veles.znicz import depooling  # pylint: disable=W0611
from veles.znicz import cutter, deconv, rprop_all2all  # pylint: disable=W0611
from veles.znicz import resizable_all2all, gd_deconv  # pylint: disable=W0611
from veles.znicz import activation
from veles.znicz.decision import DecisionsRegistry
import veles.znicz.diversity as diversity
from veles.znicz.evaluator import EvaluatorsRegistry
import veles.znicz.image_saver as image_saver
from veles.loader.base import UserLoaderRegistry, LoaderMSEMixin, CLASS_NAME
from veles.loader.image import ImageLoader
from veles.znicz.nn_rollback import NNRollback
from veles.loader.saver import MinibatchesSaver
import veles.znicz.lr_adjust as lr_adjust
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.conv import ConvolutionalBase
from veles.znicz.gd_pooling import GDPooling
from veles.znicz.all2all import All2AllSoftmax
from veles.znicz.downloader import Downloader


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
        grad_unit = grad_class(fwd.workflow, name=name, **kwargs) \
            .link_attrs(fwd, "input", "output", "weights", "bias") \
            .link_conv_attrs(fwd)
        return grad_unit

    @staticmethod
    def _create_grad_all2all(fwd, name, **kwargs):
        grad_class = GradientUnitFactory._all2all_grad_classes[type(fwd)]
        grad_unit = grad_class(fwd.workflow, name=name, **kwargs) \
            .link_attrs(fwd, "input", "output", "weights", "bias")
        return grad_unit

    @staticmethod
    def _create_grad_pooling(fwd, name, **kwargs):
        grad_class = GradientUnitFactory._pooling_grad_classes[type(fwd)]
        grad_unit = grad_class(fwd.workflow, name=name, **kwargs) \
            .link_attrs(fwd, "input", "output")
        if isinstance(fwd, pooling.MaxPooling):
            grad_unit.link_attrs(fwd, "input_offset")
        grad_unit.link_pool_attrs(fwd)
        return grad_unit

    @staticmethod
    def _create_grad_activation(fwd, name, **kwargs):
        grad_class = GradientUnitFactory._activation_grad_classes[type(fwd)]
        grad_unit = grad_class(fwd.workflow, name=name, **kwargs) \
            .link_attrs(fwd, "input", "output")
        return grad_unit

    @staticmethod
    def _create_grad_lrn(fwd, name, **kwargs):
        grad_unit = normalization.LRNormalizerBackward(
            fwd.workflow, name=name, k=fwd.k, n=fwd.n,
            alpha=fwd.alpha, beta=fwd.beta, **kwargs) \
            .link_attrs(fwd, "input", "output")
        return grad_unit

    @staticmethod
    def _create_grad_dropout(fwd, name, **kwargs):
        grad_dropout = dropout.DropoutBackward(fwd.workflow, name=name) \
            .link_attrs(fwd, "input", "output", "mask")
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


BaseWorkflowConfig = namedtuple("BaseWorkflowConfig", ("loader",))


class StandardWorkflowBase(nn_units.NNWorkflow):
    """
    A base class for standard workflows with forward and backward propagation.
    Is able to automatically create backward units by pre-created forward units

    Arguments:
        layers: list of dictionary with layers of Model
        loader_name: name of the Loader. If loader_name is None, User should \
        redefine link_loader() function and link Loader manually.
        loader_config: loader configuration parameters
    """
    WorkflowConfig = BaseWorkflowConfig
    KWATTRS = {"%s_config" % f for f in WorkflowConfig._fields}
    mcdnnic_topology_regexp = re.compile(
        "(\d+)x(\d+)x(\d+)(-(?:(\d+C\d+)|(MP\d+)|(\d+N)))*")
    mcdnnic_layer_patern = re.compile(
        "(?P<C>\d+C\d+)|(?P<MP>MP\d+)|(?P<N>\d+N)")
    mcdnnic_parse_methods = {}

    def __new__(cls, *args, **kwargs):
        if not len(cls.mcdnnic_parse_methods):
            cls.mcdnnic_parse_methods = {
                "C": cls._parse_mcdnnic_c, "N": cls._parse_mcdnnic_n,
                "MP": cls._parse_mcdnnic_mp}
        return super(StandardWorkflowBase, cls).__new__(cls)

    def __init__(self, workflow, **kwargs):
        super(StandardWorkflowBase, self).__init__(workflow, **kwargs)
        self.layer_map = nn_units.MatchingObject.mapping
        self.preprocessing = kwargs.get("preprocessing", False)
        self.mcdnnic_topology = kwargs.get("mcdnnic_topology", None)
        self.mcdnnic_parameters = kwargs.get("mcdnnic_parameters", None)
        self.layers = kwargs.get("layers", [{}])
        self._loader_name = None
        self._loader = None
        self.apply_config(**kwargs)
        if "loader_name" in kwargs:
            self.loader_name = kwargs["loader_name"]
        else:
            self.loader_factory = kwargs["loader_factory"]

    @property
    def loader_name(self):
        return self._loader_name

    @loader_name.setter
    def loader_name(self, value):
        if value is None:
            self._loader_name = value
            return
        loader_kwargs = self.dictify(self.config.loader)
        if self.mcdnnic_topology is not None:
            loader_kwargs = self._update_loader_kwargs_from_mcdnnic(
                loader_kwargs, self.mcdnnic_topology)
        self.loader_factory = UserLoaderRegistry.get_factory(
            value, **loader_kwargs)
        self._loader_name = value

    @property
    def loader_factory(self):
        return self._loader_factory

    @loader_factory.setter
    def loader_factory(self, value):
        if not callable(value):
            raise TypeError("loader_factory must be callable")
        self.loader_name = None
        self._loader_factory = value

    def reset_unit(fn):
        def wrapped(self, *args, **kwargs):
            function_name = fn.__name__
            instance_name = function_name[5:]
            self.unlink_unit(instance_name)
            return fn(self, *args, **kwargs)

        return wrapped

    def check_forward_units(fn):
        def wrapped(self, *args, **kwargs):
            self._check_forwards()
            return fn(self, *args, **kwargs)

        return wrapped

    def check_backward_units(fn):
        def wrapped(self, *args, **kwargs):
            self._check_gds()
            return fn(self, *args, **kwargs)

        return wrapped

    def unlink_unit(self, remove_unit_name):
        if hasattr(self, remove_unit_name):
            self.warning(
                "Instance %s exists. It will be removed and unlink"
                % remove_unit_name)
            remove_unit = getattr(self, remove_unit_name)
            remove_unit.unlink_all()
            self.del_ref(remove_unit)

    def apply_config(self, **kwargs):
        old_config = getattr(self, "config", None)
        self.config = self.WorkflowConfig(
            **{f: self.config2kwargs(kwargs.pop("%s_config" % f,
                                                getattr(old_config, f, {})))
               for f in self.WorkflowConfig._fields})

    @property
    def mcdnnic_topology(self):
        return self._mcdnnic_topology

    @mcdnnic_topology.setter
    def mcdnnic_topology(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("mcdnnic_topology must be a string")
            if not self.mcdnnic_topology_regexp.match(value):
                raise ValueError(
                    "mcdnnic_topology value must match the following regular"
                    "expression: %s (got %s)"
                    % (self.mcdnnic_topology_regexp.pattern, value))
        self._mcdnnic_topology = value

    @property
    def layers(self):
        if self.mcdnnic_topology is not None:
            return self._get_layers_from_mcdnnic(
                self.mcdnnic_topology)
        else:
            return self._layers

    @layers.setter
    def layers(self, value):
        if self.mcdnnic_topology is not None and value != [{}]:
            raise ValueError(
                "Please do not set mcdnnic_topology and layers at the same "
                "time.")
        if not isinstance(value, list):
            raise ValueError("layers should be a list of dicts")
        if (value == [{}] and self.mcdnnic_topology is None
                and not self.preprocessing):
            raise error.BadFormatError(
                "Looks like layers is empty and mcdnnic_topology is not "
                "defined. Please set layers like in VELES samples or"
                "mcdnnic_topology like in artical 'Multi-column Deep Neural"
                "Networks for Image Classification'"
                "(http://papers.nips.cc/paper/4824-imagenet-classification-wi"
                "th-deep-convolutional-neural-networks)")
        for layer in value:
            if not isinstance(layer, dict):
                raise ValueError(
                    "layers should be a list of dicts")
        self._layers = value

    @property
    def preprocessing(self):
        return self._preprocessing

    @preprocessing.setter
    def preprocessing(self, value):
        self._preprocessing = value

    def _get_mcdnnic_parameters(self, arrow):
        if (self.mcdnnic_parameters is not None
                and arrow in self.mcdnnic_parameters):
            return self.mcdnnic_parameters[arrow]
        else:
            return {}

    @staticmethod
    def _parse_mcdnnic_c(index, value):
        kernels, kx = value.split("C")
        return {
            "type": "conv",
            "->": {"n_kernels": int(kernels), "kx": int(kx), "ky": int(kx)}}

    @staticmethod
    def _parse_mcdnnic_mp(index, value):
        _, kx = value.split("MP")
        return {"type": "max_pooling", "->": {"kx": int(kx), "ky": int(kx)}}

    @staticmethod
    def _parse_mcdnnic_n(index, value):
        neurons, _ = value.split("N")
        if index:
            return {"type": "softmax",
                    "->": {"output_sample_shape": int(neurons)}}
        else:
            return {"type": "all2all",
                    "->": {"output_sample_shape": int(neurons)}}

    def _get_layers_from_mcdnnic(self, description):
        layers = []
        forward_parameters = self._get_mcdnnic_parameters("->")
        backward_parameters = self._get_mcdnnic_parameters("<-")
        matches = tuple(re.finditer(self.mcdnnic_layer_patern, description))
        for index, match in enumerate(matches):
            match_name = next(n for n, v in match.groupdict().items() if v)
            layer_config = self.mcdnnic_parse_methods[match_name](
                index == len(matches) - 1, match.group(match_name))
            layer_config["->"].update(forward_parameters)
            layer_config["<-"] = backward_parameters
            layers.append(layer_config)
        return layers

    def _update_loader_kwargs_from_mcdnnic(self, kwargs, description):
        inp = description.split("-")[0]
        minibatch_size, y_size, x_size = inp.split("x")
        kwargs["minibatch_size"] = int(minibatch_size)
        kwargs["scale"] = (int(y_size), int(x_size))
        return kwargs

    def link_forwards(self, init_attrs, *parents):
        """
        Creates forward units ( :class:`veles.znicz.nn_units.ForwardBase`
        descendant) from "layers" configuration.
        Links first forward unit from \*parents argument.
        Links init_attrs argument with first forward unit attributes.
        For each layer adds a new forward unit to self.forwards, links it with
        the previous forward unit by :func:`veles.units.Unit.link_from()` .
        Links attributes of that unit with attributes of the previous forward
        unit by :func:`veles.units.Unit.link_attrs()` .
        Returns the last of :class:`veles.znicz.nn_units.ForwardBase`
        descendant units.

        Arguments:
            init_attrs: attrubutes of parents unit, which will be transfer to\
            first forward unit
            parents: units, from whom will be link first forward unit
        """
        del self.forwards[:]
        for _i, layer in enumerate(self.layers):
            tpe, kwargs, _ = self._get_layer_type_kwargs(layer)
            try:
                unit = self.layer_map[tpe].forward(self, **kwargs)
            except IndexError:
                raise from_none(ValueError("Failed to find a Forward in %s" %
                                           tpe))
            self._add_forward_unit(unit, init_attrs, *parents)
        # Another loop for ZeroFiller unit. Linking attributes for
        # ZeroFiller from attributes of next layer
        for prev_forward, forward in zip(self.forwards, self.forwards[1:]):
            if isinstance(prev_forward, weights_zerofilling.ZeroFiller):
                prev_forward.link_attrs(forward, "weights")

        last_fwd = self.forwards[-1]
        if not isinstance(last_fwd, All2AllSoftmax) and \
                not isinstance(self.real_loader, LoaderMSEMixin):
            return last_fwd

        def on_initialized():
            import veles

            if isinstance(self.real_loader, veles.loader.base.LoaderMSEMixin):
                if (last_fwd.output_sample_shape != tuple() and
                        numpy.prod(last_fwd.output_sample_shape)
                        != numpy.prod(self.real_loader.targets_shape)):
                    self.warning("Overriding %s.output_sample_shape with %s",
                                 last_fwd, self.real_loader.targets_shape)
                else:
                    self.info("Setting %s.output_sample_shape to %s",
                              last_fwd, self.real_loader.targets_shape)
                last_fwd.output_sample_shape = self.real_loader.targets_shape
            elif isinstance(last_fwd, veles.znicz.all2all.All2AllSoftmax):
                ulc = self.real_loader.unique_labels_count
                oss = last_fwd.output_sample_shape
                if oss != tuple() and numpy.prod(oss) != ulc:
                    self.warning(
                        "Overriding %s.output_sample_shape %s with (%s,)",
                        last_fwd, oss, ulc)
                else:
                    self.info("Setting %s.output_sample_shape to %d",
                              last_fwd, ulc)
                last_fwd.output_sample_shape = ulc

        self.real_loader.on_initialized = on_initialized
        return last_fwd

    def link_repeater(self, *parents):
        """
        Links :class:`veles.workflow.Repeater` instance from \*parents.
        Returns :class:`veles.workflow.Repeater` instance.

        Arguments:
            parents: units to link this one from.
        """
        self.repeater.link_from(*parents)
        return self.repeater

    def link_fire_starter(self, *parents):
        """
        Links :class:`veles.plumbing.FireStarter` instance from \*parents.
        Returns :class:`veles.plumbing.FireStarter` instance.

        Arguments:
            parents: units to link this one from.
        """
        self.fire_starter = FireStarter(self)
        self.fire_starter.link_from(*parents)
        return self.fire_starter

    def dictify(self, obj):
        return getattr(obj, "__content__", obj)

    def config2kwargs(self, unit_config):
        return {} if unit_config is None else self.dictify(unit_config)

    def link_loader(self, *parents):
        """
        Creates a new :class:`veles.loader.base.Loader` descendant. The actual
        class type is taken from the global mapping by "loader_name" key.
        Links :class:`veles.loader.base.Loader` descendant from \*parents.
        Returns :class:`veles.loader.base.Loader` descendant instance.

        Arguments:
            parents: units to link this one from.
        """
        self.loader = self.loader_factory(self)  # pylint: disable=E1102
        self.loader.link_from(*parents)
        # Save this loader, since it can be later replaced with an Avatar
        self.real_loader = self.loader
        return self.loader

    def link_end_point(self, *parents):
        """
        Links the existing :class:`veles.workflow.EndPoint` and
        :class:`veles.workflow.Repeater` with \*parents.
        Returns :class:`veles.workflow.EndPoint` instance.

        Arguments:
            parents: units to link this one from.
        """
        self.repeater.link_from(*parents)
        self.end_point.link_from(*parents)
        return self.end_point

    def create_workflow(self):
        self.link_repeater(self.start_point)

        self.link_loader(self.repeater)

        # Add forwards units
        self.link_forwards(("input", "minibatch_data"), self.loader)

        self.end_point.gate_block = ~self.loader.complete

    def _get_layer_type_kwargs(self, layer):
        tpe = layer.get("type", "").strip()
        if not tpe:
            raise ValueError("layer type must not be an empty string")
        if tpe not in self.layer_map:
            raise ValueError("Unknown layer type %s" % tpe)
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

    def _add_forward_unit(self, new_unit, init_attrs=None, *parents):
        """
        Adds a new forward unit to self.forwards, links it with previous fwd
        unit by link_from and link_attrs. If self.forwards is empty, links unit
        with new_unit
        """
        if len(self.forwards) > 0:
            prev_forward_unit = self.forwards[-1],
        else:
            if len(parents) == 0:
                raise ValueError(
                    "No parent units were specified for the first forward!")
            prev_forward_unit = parents

        new_unit.link_from(*prev_forward_unit)

        self.forwards.append(new_unit)

        if not hasattr(new_unit, "input"):
            return

        for fwd in reversed(self.forwards[:-1]):
            if hasattr(fwd, "output"):
                new_unit.link_attrs(fwd, ("input", "output"))
                break
        else:
            new_unit.link_attrs(parents[0], init_attrs)

    reset_unit = staticmethod(reset_unit)
    check_forward_units = staticmethod(check_forward_units)
    check_backward_units = staticmethod(check_backward_units)


StandardWorkflowConfig = namedtuple(
    "StandardWorkflowConfig",
    ("decision", "snapshotter", "image_saver", "evaluator", "data_saver",
     "result_loader", "weights_plotter", "similar_weights_plotter",
     "lr_adjuster", "downloader", "publisher", "rollback")
    + BaseWorkflowConfig._fields)


class StandardWorkflow(StandardWorkflowBase):
    """
    Workflow for trivially connections between Unit.
    User can create Self-constructing Models with that class.
    It means that User can change structure of Model (Convolutional,
    Fully connected, different parameters) and parameters of training in
    configuration file.

    Arguments:
        loss_function: name of Loss function. Choices are "softmax" or "mse"
        decision_name: name of Decision. If loss_function was defined and \
        decision_name was not, decision_name creates automaticly
        evaluator_name: name of Evaluator. If loss_function was defined and \
        evaluator_name was not, evaluator_name creates automaticly
        decision_config: decision configuration parameters
        snapshotter_config: snapshotter configuration parameters
        image_save_configr: image_saver configuration parameters
        data_saver_config: data_saver configuration parameters
        result_loader_config: result_loader configuration parameters
        similar_weights_plotter_config: similar_weights_plotter configuration\
        parameters
        result_loader_name: The forward workflow's loader name. Not neccessary\
        if forward workflow is not going to be extracted.
        result_unit_factory: The results' publishing unit factory.
    """
    WorkflowConfig = StandardWorkflowConfig
    CONFIGURABLE_UNIT_NAMES = "result_loader", "decision", "evaluator", \
                              "snapshotter"
    KWATTRS = {"%s_config" % f for f in WorkflowConfig._fields}.union(
        {"%s_name" % n for n in CONFIGURABLE_UNIT_NAMES})

    def __init__(self, workflow, **kwargs):
        super(StandardWorkflow, self).__init__(workflow, **kwargs)
        self.result_unit_factory = kwargs.get("result_unit_factory")
        self.loss_function = kwargs.get("loss_function", None)
        for unit_name in self.CONFIGURABLE_UNIT_NAMES:
            setattr(self, "%s_name" % unit_name,
                    kwargs.pop("%s_name" % unit_name, None))

        self.create_workflow()

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        if value not in ("softmax", "mse", None):
            raise ValueError("Unknown loss function type %s" % value)
        self._loss_function = value

    def _set_name_of_unit(self, value, name, mapping):
        value_error = "%s name or loss function must be defined" % name
        if (value is None and self.loss_function is None and
                not self.preprocessing):
            raise ValueError(value_error)
        setattr(self, "_%s_name" % name, value)
        if value is None and self.loss_function is not None:
            setattr(self, "_%s_name" % name, mapping[self.loss_function])
        if value is not None:
            setattr(self, "_%s_name" % name, value)
            if self.loss_function is not None:
                self.warning("Loss function and %s name is defined at the "
                             "same time. %s name has higher priority, then "
                             "loss function" % (name, name))

    @property
    def decision_name(self):
        return self._decision_name

    @decision_name.setter
    def decision_name(self, value):
        self._set_name_of_unit(
            value, "decision", DecisionsRegistry.loss_mapping)

    @property
    def evaluator_name(self):
        return self._evaluator_name

    @evaluator_name.setter
    def evaluator_name(self, value):
        self._set_name_of_unit(
            value, "evaluator", EvaluatorsRegistry.loss_mapping)

    def link_forwards(self, init_attrs, *parents):
        last_fwd = super(StandardWorkflow, self).link_forwards(
            init_attrs, *parents)
        if self.loss_function == "mse" and \
                isinstance(last_fwd, All2AllSoftmax):
            raise NotImplementedError(
                "Softmax last layer does not currently support MSE.")

    def create_workflow(self):
        # Add repeater unit
        self.link_repeater(self.start_point)

        # Add loader unit
        self.link_loader(self.repeater)

        # Add forwards units
        self.link_forwards(("input", "minibatch_data"), self.loader)

        # Add evaluator unit
        self.link_evaluator(self.forwards[-1])

        # Add decision unit
        self.link_decision(self.evaluator)

        # Add snapshotter unit
        self.link_snapshotter(self.decision)

        # Add gradient descent units
        last_gd = self.link_gds(self.snapshotter)

        # Add error or mse plotter unit
        if self.loss_function == "mse":
            last_err = self.link_mse_plotter(last_gd)
        else:
            last_err = self.link_min_max_plotter(
                self.link_error_plotter(last_gd))

        # Loop the workflow
        self.link_loop(last_err)

        # Add end_point unit
        self.link_end_point(last_gd)

    def extract_forward_workflow(self, loader_unit_factory=None,
                                 loader_name=None, loader_config=None,
                                 result_unit_factory=None,
                                 result_unit_config=None, cyclic=True):
        """
        Generates a separate forward propagation workflow from this one,
        taking the trained weights, settings, etc.
        :param loader_unit_factory: callable(workflow) which returns the
        loader unit.
        :param loader_name: Alternative to loader_unit_factory, loader name in
          UserLoaderRegistry.
        :param loader_config: Used in pair with loader_name to configure the
        loader. May be a dictionary or an instance of veles.config.Config.
        :param result_unit_factory: callable(workflow) which returns the result
        output unit.
        :param result_unit_config: Passed into result_unit_factory as keyword
        arguments. May be a dictionary or an instance of veles.config.Config.
        :param cyclic: True if the loader decides whether to stop the workflow;
        otherwise, False => the extracted workflow is going to do a single
        iteration.
        :return: veles.znicz.standard_workflow.StandardWorkflowBase instance.
        """
        self.debug("Constructing the new workflow...")
        if loader_unit_factory is not None:
            assert loader_name is None and loader_config is None
            wf = StandardWorkflowBase(self.workflow,
                                      name="Forwards@%s" % self.name,
                                      loader_factory=loader_unit_factory,
                                      layers=self.layers)
        else:
            wf = StandardWorkflowBase(self.workflow,
                                      name="Forwards@%s" % self.name,
                                      loader_name=loader_name,
                                      loader_config=loader_config,
                                      layers=self.layers)
        if cyclic:
            start_unit = wf.link_repeater(wf.start_point)
        else:
            start_unit = wf.start_point
        wf.link_loader(start_unit)
        wf.loader.derive_from(self.loader)
        if cyclic:
            assert hasattr(wf.loader, "complete"), \
                "The specified loader does not have \"complete\" flag."
            wf.end_point.link_from(wf.loader).gate_block = ~wf.loader.complete
        wf.link_forwards(("input", "minibatch_data"), wf.loader)
        if cyclic:
            wf.forwards[0].gate_block = wf.loader.complete
        result_unit_config = self.config2kwargs(result_unit_config)
        wf.result_unit = result_unit_factory(wf, **result_unit_config) \
            .link_from(wf.forwards[-1])
        wf.result_unit.link_attrs(wf.forwards[-1], ("input", "output"))
        wf.result_unit.link_attrs(
            wf.loader, ("labels_mapping", "reversed_labels_mapping"))
        if self.loss_function == "mse":
            wf.result_unit.link_attrs(wf.loader, "target_normalizer")
        if cyclic:
            wf.repeater.link_from(wf.result_unit)
        else:
            wf.link_end_point(wf.result_unit)
        self.debug("Importing forwards...")
        for fwd_exp, fwd_imp in zip(self.forwards, wf.forwards):
            fwd_imp.apply_data_from_master(
                fwd_exp.generate_data_for_slave(None))
        return wf

    @StandardWorkflowBase.check_forward_units
    def link_gds(self, *parents):
        """
        Creates :class:`veles.znicz.nn_units.GradientDescentBase`
        descendant units from from "layers" configuration.
        Link the last of :class:`veles.znicz.nn_units.GradientDescentBase`
        descendant units from \*parents.
        Links attributes of the last
        :class:`veles.znicz.nn_units.GradientDescentBase` descendant units
        from :class:`veles.znicz.evaluator.EvaluatorBase` descendant,
        :class:`veles.znicz.decision.DecisionBase` descendant and corresponded
        :class:`veles.znicz.nn_units.ForwardBase` descendant unit.
        Links :class:`veles.znicz.nn_units.GradientDescentBase`
        descendant with previous
        :class:`veles.znicz.nn_units.GradientDescentBase` descendant in gds.
        Links attributes of :class:`veles.znicz.nn_units.GradientDescentBase`
        descendant from previous
        :class:`veles.znicz.nn_units.GradientDescentBase` descendant,
        :class:`veles.znicz.decision.DecisionBase` descendant and
        corresponded :class:`veles.znicz.nn_units.ForwardBase` descendant unit.
        Returns the first :class:`veles.znicz.nn_units.GradientDescentBase`
        which correspond to the first :class:`veles.znicz.nn_units.ForwardBase`
        descendant (but the first
        :class:`veles.znicz.nn_units.GradientDescentBase` runs the last of all
        gds. Do not be confused).
        Arguments:
            parents: units, from whom will be link last of\
            :class:`veles.znicz.nn_units.GradientDescentBase` descendant units
        """
        if type(self.layers) != list:
            raise error.BadFormatError("layers should be a list of dicts")
        self.gds[:] = (None,) * len(self.layers)
        first_gd = None
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
            if first_gd is not None:
                unit.link_from(first_gd) \
                    .link_attrs(first_gd, ("err_output", "err_input"))
            else:
                unit.link_from(*parents) \
                    .link_attrs(self.evaluator, "err_output")
            first_gd = unit

            attrs = []
            # TODO(v.markovtsev): add "wants" to Unit and use it here
            try_link_attrs = {"input", "weights", "bias", "input_offset",
                              "mask", "output"}
            if isinstance(unit, ConvolutionalBase):
                try_link_attrs.update(ConvolutionalBase.CONV_ATTRS)
            if isinstance(unit, GDPooling):
                try_link_attrs.update(GDPooling.POOL_ATTRS)
            for attr in try_link_attrs:
                if hasattr(self.forwards[i], attr):
                    attrs.append(attr)
            unit.link_attrs(self.forwards[i], *attrs)

            unit.gate_skip = self.decision.gd_skip

        # Remove None elements
        for i in units_to_delete:
            del self.gds[i]

        # Disable error backpropagation on the last layer
        self.gds[0].need_err_input = False

        return first_gd

    def link_loop(self, parent):
        """
        Closes the loop based on the :class:`veles.workflow.Repeater`.

        Arguments:
            parent: unit, from whom will be link\
            :class:`veles.workflow.Repeater` unit
        """
        self.repeater.link_from(parent)

    def link_avatar(self, *extra_attrs):
        """
        Replaces the current loader with it's avatar, allowing the parallel
        work of the loader and the main contour.
        Please note that the loader must be linked from the start point, not
        the repeater.
        :param extra_attrs: Additional attributes to copy from the loader.
        :return: The linked :class:`veles.avatar.Avatar` unit.
        """
        self.loader.ignores_gate <<= True
        self.avatar = Avatar(self)
        self.avatar.reals[self.loader] = self.loader.exports + extra_attrs
        self.avatar.clone()
        self.avatar.link_from(self.loader)
        self.loader.link_from(self.avatar)
        self.avatar.link_from(
            self.repeater).gate_block = self.loader.gate_block
        self.loader = self.avatar
        return self.avatar

    @StandardWorkflowBase.reset_unit
    def link_downloader(self, *parents):
        self.downloader = Downloader(self, **self.config.downloader)
        self.downloader.link_from(*parents)

    @StandardWorkflowBase.reset_unit
    @StandardWorkflowBase.check_forward_units
    def link_evaluator(self, *parents):
        """
        Creates instance of :class:`veles.znicz.evaluator.EvaluatorBase`
        descendant unit given the "loss_function" parameter.
        Links :class:`veles.znicz.evaluator.EvaluatorBase`
        descendant unit from \*parents.
        Links attributes of :class:`veles.znicz.evaluator.EvaluatorBase`
        descendant unit from attributes of :class:`veles.loader.base.Loader`
        descendant and :class:`veles.znicz.nn_units.ForwardBase` descendant.
        Returns instance of :class:`veles.znicz.evaluator.EvaluatorBase`
        descendant unit.

        Arguments:
            parents: units to link this one from.
            :class:`veles.znicz.evaluator.EvaluatorBase` descendant unit
        """
        self.evaluator = EvaluatorsRegistry.evaluators[
            self.evaluator_name](self, **self.config.evaluator) \
            .link_from(*parents).link_attrs(self.forwards[-1], "output") \
            .link_attrs(self.loader,
                        ("batch_size", "minibatch_size"),
                        ("labels", "minibatch_labels"),
                        ("max_samples_per_epoch", "total_samples"),
                        "class_lengths", ("offset", "global_offset"))
        if self.evaluator_name == "evaluator_softmax":
            self.evaluator.link_attrs(self.forwards[-1], "max_idx")
        elif self.evaluator_name == "evaluator_mse":
            self.evaluator.link_attrs(
                self.loader, ("target", "minibatch_targets"),
                "class_targets", ("normalizer", "target_normalizer"))
        return self.evaluator

    @StandardWorkflowBase.reset_unit
    def link_decision(self, *parents):
        """
        Creates instance of :class:`veles.znicz.decision.DecisionBase`
        descendant unit given the "loss_function" parameter.
        Links :class:`veles.znicz.decision.DecisionBase`
        descendant unit from \*parents.
        Links attributes of :class:`veles.znicz.decision.DecisionBase`
        descendant from attributes of :class:`veles.loader.base.Loader`
        descendant, :class:`veles.znicz.evaluator.EvaluatorBase` descendant,
        :class:`veles.znicz.decision.DecisionBase` descendant,
        :class:`veles.workflow.Repeater`.
        Returns instance of :class:`veles.znicz.decision.DecisionBase`
        descendant.

        Arguments:
            parents: units to link this one from.
            :class:`veles.znicz.decision.DecisionBase` descendant unit
        """
        self.decision = DecisionsRegistry.decisions[
            self.decision_name](self, **self.config.decision) \
            .link_from(*parents) \
            .link_attrs(self.loader, "minibatch_class", "last_minibatch",
                        "minibatch_size", "class_lengths", "epoch_ended",
                        "epoch_number")
        if self.decision_name == "decision_mse":
            self.decision.link_attrs(self.loader, "minibatch_offset")
        self.decision.link_attrs(self.evaluator, ("minibatch_n_err", "n_err"))
        if self.decision_name == "decision_gd":
            self.decision.link_attrs(
                self.evaluator,
                ("minibatch_confusion_matrix", "confusion_matrix"),
                ("minibatch_max_err_y_sum", "max_err_output_sum"))
        elif self.decision_name == "decision_mse":
            self.decision.link_attrs(
                self.evaluator,
                ("minibatch_metrics", "metrics"),
                ("minibatch_mse", "mse"))
        self.repeater.gate_block = self.decision.complete
        self.real_loader.gate_block = self.decision.complete
        return self.decision

    @StandardWorkflowBase.reset_unit
    def link_snapshotter(self, *parents):
        """
        Creates instance of :class:`veles.snapshotter.SnapshotterBase`
        descendant unit.
        Links :class:`veles.snapshotter.SnapshotterBase`
        descendant unit from \*parents.
        Links attributes of :class:`veles.snapshotter.SnapshotterBase`
        descendant from attributes of
        :class:`veles.znicz.decision.DecisionBase` descendant.
        Returns instance of :class:`veles.snapshotter.SnapshotterBase`
        descendant.

        Arguments:
            parents: units to link this one from.
            :class:`veles.snapshotter.SnapshotterBase` descendant unit
        """
        snapshotter_name = self.snapshotter_name or "nnfile"
        self.snapshotter = SnapshotterRegistry.snapshotters[snapshotter_name](
            self, **self.config.snapshotter) \
            .link_from(*parents) \
            .link_attrs(self.decision, ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = ~self.decision.epoch_ended
        self.snapshotter.skip = ~self.decision.improved
        return self.snapshotter

    def link_end_point(self, *parents):
        """
        Links the existing :class:`veles.workflow.EndPoint` unit with
        \*parents. Returns :class:`veles.workflow.EndPoint` instance.

        Arguments:
            parents: units to link this one from.
            :class:`veles.workflow.EndPoint` unit
        """
        self.end_point.link_from(*parents)
        self.end_point.gate_block = ~self.decision.complete
        return self.end_point

    @StandardWorkflowBase.reset_unit
    @StandardWorkflowBase.check_forward_units
    def link_image_saver(self, *parents):
        """
        Creates instance of :class:`veles.znicz.image_saver.ImageSaver` .
        Links :class:`veles.znicz.image_saver.ImageSaver` unit with \*parents.
        Links attributes of :class:`veles.znicz.image_saver.ImageSaver` with
        attributes of :class:`veles.znicz.nn_units.ForwardBase`
        descendant, :class:`veles.loader.base.Loader` descendant,
        :class:`veles.znicz.decision.DecisionBase` descendant and
        :class:`veles.snapshotter.SnapshotterBase` descendant units.
        Returns instance of :class:`veles.znicz.image_saver.ImageSaver`.

        Arguments:
            parents: units to link this one from.
            :class:`veles.znicz.image_saver.ImageSaver` unit
        """
        self.image_saver = \
            image_saver.ImageSaver(self, **self.config.image_saver) \
            .link_from(*parents)
        if self.evaluator_name == "evaluator_softmax":
            self.image_saver.link_attrs(self.forwards[-1], "max_idx")
        self.image_saver.link_attrs(self.forwards[-1], "output")
        if isinstance(self.loader, ImageLoader):
            self.image_saver.link_attrs(self.loader, "color_space")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indices", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")
        if self.evaluator_name == "evaluator_mse":
            self.image_saver.link_attrs(
                self.loader, ("target", "minibatch_targets"))
        self.image_saver.link_attrs(self.snapshotter,
                                    ("this_save_time", "time")) \
            .gate_skip = ~self.decision.improved
        return self.image_saver

    @StandardWorkflowBase.reset_unit
    @StandardWorkflowBase.check_backward_units
    def link_lr_adjuster(self, *parents):
        """
        Creates instance of :class:`veles.znicz.lr_adjust.LearningRateAdjust`
        unit.
        Links :class:`veles.znicz.lr_adjust.LearningRateAdjust` unit with
        \*parents. Changing "learning_rate" and "learning_rate_bias" in
        :class:`veles.znicz.nn_units.GradientDescentBase` descendant units.
        Returns instance of :class:`veles.znicz.lr_adjust.LearningRateAdjust`.

        Arguments:
            parents: units to link this one from.
            :class:`veles.znicz.lr_adjust.LearningRateAdjust` unit
        """
        self.lr_adjuster = lr_adjust.LearningRateAdjust(
            self, **self.dictify(self.config.lr_adjuster))
        for gd_elm in self.gds:
            self.lr_adjuster.add_gd_unit(gd_elm)
        self.lr_adjuster.link_from(*parents)
        return self.lr_adjuster

    @StandardWorkflowBase.reset_unit
    def link_rollback(self, *parents):
        self.rollback = NNRollback(self, **self.config.rollback)
        self.rollback.link_from(*parents)
        self.rollback.improved = self.decision.train_improved
        self.rollback.gate_skip = ~self.loader.epoch_ended | \
            self.decision.complete
        return self.rollback

    @StandardWorkflowBase.reset_unit
    def link_meandispnorm(self, *parents):
        """
        Creates an instance of
        :class:`veles.mean_disp_normalizer.MeanDispNormalizer` unit.
        Links :class:`veles.mean_disp_normalizer.MeanDispNormalizer`
        unit with \*parents.
        Links attributes of
        :class:`veles.mean_disp_normalizer.MeanDispNormalizer` with
        attributes of :class:`veles.loader.base.Loader` descendant.
        Returns instance of
        :class:`veles.mean_disp_normalizer.MeanDispNormalizer`.

        Arguments:
            parents: units to link this one from.
            :class:`veles.mean_disp_normalizer.MeanDispNormalizer` unit
        """
        self.meandispnorm = MeanDispNormalizer(self) \
            .link_attrs(self.loader, ("input", "minibatch_data"),
                        "mean", "rdisp") \
            .link_from(*parents)
        return self.meandispnorm

    @StandardWorkflowBase.reset_unit
    def link_gd_diff_stats(self, *parents, **kwargs):
        """
        Creates an instance of
        :class:`veles.znicz.diff_stats.DiffStatsr` unit.
        Links :class:`veles.znicz.diff_stats.DiffStats` unit with \*parents.
        Links attributes of
        :class:`veles.znicz.diff_stats.DiffStats` with attributes of
        gradient descent units.

        :param parents: units to link this one from.
        :param file_name: file name with the results.
        :return: instance of :class:`veles.znicz.diff_stats.DiffStats`.
        """
        file_name = kwargs.get("file_name",
                               "diff_stats.%d.pickle" % best_protocol)
        self.gd_diff_stats = DiffStats(
            self, arrays={u: ("gradient_weights",) for u in self.gds},
            file_name=file_name).link_from(*parents)
        self.gd_diff_stats.gate_skip = self.decision.gd_skip
        return self.gd_diff_stats

    @StandardWorkflowBase.reset_unit
    def link_ipython(self, *parents):
        """
        Creates instance of :class:`veles.interaction.Shell` unit.
        Links :class:`veles.interaction.Shell`  unit with \*parents.
        Returns instance of :class:`veles.interaction.Shell`.

        Arguments:
            parents: units to link this one from.
            :class:`veles.interaction.Shell` unit
        """
        self.ipython = Shell(self).link_from(*parents) \
            .gate_skip = ~self.decision.epoch_ended
        return self.ipython

    @StandardWorkflowBase.reset_unit
    def link_publisher(self, *parents):
        self.publisher = Publisher(self, **self.config.publisher) \
            .link_from(*parents)
        self.publisher.result_providers.add(self.decision)
        self.publisher.loader_unit = self.real_loader
        self.publisher.gate_skip = ~self.decision.complete
        return self.publisher

    @StandardWorkflowBase.reset_unit
    def link_error_plotter(self, *parents):
        """
        Creates the list of instances of
        :class:`veles.plotting_units.AccumulatingPlotter` units.
        Links the first :class:`veles.plotting_units.AccumulatingPlotter` unit
        with \*parents.
        Links each :class:`veles.plotting_units.AccumulatingPlotter` unit from
        previous :class:`veles.plotting_units.AccumulatingPlotter` unit.
        Links attributes of :class:`veles.plotting_units.AccumulatingPlotter`
        units from attributes of :class:`veles.znicz.decision.DecisionBase`
        descendant (epoch_n_err_pt).
        Returns the last of :class:`veles.plotting_units.AccumulatingPlotter`
        units.

        Arguments:
            parents: units, from whom will be link the first of\
            :class:`veles.plotting_units.AccumulatingPlotter` units.
        """
        self.error_plotters = []
        prev = parents
        styles = ["r-", "b-", "k-"]
        for i in 1, 2:
            plotter = plotting_units.AccumulatingPlotter(
                self, name="Number of errors", plot_style=styles[i],
                label=CLASS_NAME[i]) \
                .link_attrs(self.decision, ("input", "epoch_n_err_pt")) \
                .link_from(*prev)
            plotter.input_field = i
            plotter.gate_skip = ~self.decision.epoch_ended
            self.error_plotters.append(plotter)
            prev = plotter,
        self.error_plotters[0].clear_plot = True
        self.error_plotters[-1].redraw_plot = True
        return prev[0]

    def link_conf_matrix_plotter(self, *parents):
        """
        Creates the list of instances of
        :class:`veles.plotting_units.MatrixPlotter`.
        Links the first :class:`veles.plotting_units.MatrixPlotter` unit
        with \*parents.
        Links each :class:`veles.plotting_units.MatrixPlotter` unit from
        previous :class:`veles.plotting_units.MatrixPlotter` unit.
        Links attributes of :class:`veles.plotting_units.MatrixPlotter` units
        from attributes of :class:`veles.znicz.decision.DecisionBase`
        cd descendant.
        Returns the last of :class:`veles.plotting_units.MatrixPlotter` units.

        Arguments:
            parents: units, from whom will be link the first of\
            :class:`veles.plotting_units.MatrixPlotter` units.
        """
        self.conf_matrix_plotters = []
        prev = parents
        for i in range(1, len(self.decision.confusion_matrixes)):
            mp = plotting_units.MatrixPlotter(
                self, name=(CLASS_NAME[i] + " matrix")) \
                .link_attrs(self.decision, ("input", "confusion_matrixes")) \
                .link_from(*prev)
            mp.input_field = i
            mp.link_attrs(self.loader, "reversed_labels_mapping")
            mp.gate_skip = ~self.decision.epoch_ended
            self.conf_matrix_plotters.append(mp)
            prev = mp,
        return prev[0]

    def link_err_y_plotter(self, *parents):
        """
        Creates the list of instances of
        :class:`veles.plotting_units.AccumulatingPlotter`.
        Links the first :class:`veles.plotting_units.AccumulatingPlotter` unit
        with \*parents.
        Links each :class:`veles.plotting_units.AccumulatingPlotter` unit from
        previous :class:`veles.plotting_units.AccumulatingPlotter` unit.
        Links attributes of :class:`veles.plotting_units.AccumulatingPlotter`
        units from attributes of :class:`veles.znicz.decision.DecisionBase`
        descendant (max_err_y_sums).
        Returns the last instance of
        :class:`veles.plotting_units.AccumulatingPlotter`.

        Arguments:
            parents: units, from whom will be link the first of\
            :class:`veles.plotting_units.AccumulatingPlotter` units.
        """
        styles = ["r-", "b-", "k-"]
        self.err_y_plotters = []
        prev = parents
        for i in 1, 2:
            plotter = plotting_units.AccumulatingPlotter(
                self, name="Last layer max gradient sum",
                plot_style=styles[i],
                label=("Test", "Validation", "Train")[i]).link_attrs(
                self.decision, ("input", "max_err_y_sums")).link_from(*prev)
            plotter.input_field = i
            plotter.gate_skip = ~self.decision.epoch_ended
            self.err_y_plotters.append(plotter)
            prev = plotter,
        self.err_y_plotters[0].clear_plot = True
        self.err_y_plotters[-1].redraw_plot = True
        return prev[0]

    def link_multi_hist_plotter(self, weights_input, *parents):
        """
        Creates the list of instances of
        :class:`veles.plotting_units.MultiHistogram` units.
        Links the first :class:`veles.plotting_units.MultiHistogram` unit
        with \*parents.
        Links each :class:`veles.plotting_units.MultiHistogram` unit from
        previous :class:`veles.plotting_units.MultiHistogram` unit.
        Links attributes of :class:`veles.plotting_units.MultiHistogram` units
        from attributes of :class:`veles.znicz.decision.DecisionBase`
        descendant.
        Returns the last :class:`veles.plotting_units.MultiHistogram` unit.

        Arguments:
            weights_input: weights to plotting. "weights" or\
            "gradient_weights" for example
            limit: max number of pictures on one plotter
            parents: units, from whom will be link the first of\
            :class:`veles.plotting_units.MultiHistogram` units.
        """

        self.multi_hist_plotter = []
        prev = parents
        link_units = self._get_weights_source_units(weights_input)
        index = 1
        for i, layer in enumerate(self.layers):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All) or
                    isinstance(self.forwards[i], all2all.All2AllSoftmax)):
                continue
            multi_hist = plotting_units.MultiHistogram(
                self, name="Histogram #%s: %s" % (index, layer["type"])) \
                .link_from(*prev).link_attrs(
                    link_units[i], ("input", weights_input))
            multi_hist.gate_skip = ~self.decision.epoch_ended
            index += 1
            prev = multi_hist,
            self.multi_hist_plotter.append(multi_hist)
            for name in "n_kernels", "output_sample_shape":
                if layer.get(name) is not None:
                    multi_hist.hist_number = layer[name]
                    break
        return prev[0]

    @StandardWorkflowBase.check_forward_units
    def link_weights_plotter(self, weights_input, *parents):
        """
        Creates the list of instances of
        :class:`veles.znicz.nn_plotting_units.Weights2D` units.
        Links the first :class:`veles.znicz.nn_plotting_units.Weights2D` unit
        with \*parents.
        Links each :class:`veles.znicz.nn_plotting_units.Weights2D` unit from
        previous :class:`veles.znicz.nn_plotting_units.Weights2D` unit.
        Links attributes of :class:`veles.znicz.nn_plotting_units.Weights2D`
        units from attributes of :class:`veles.znicz.decision.DecisionBase`
        descendant, :class:`veles.loader.base.Loader` descendant,
        :class:`veles.znicz.nn_units.ForwardBase` descendant.
        Returns the last instance of
        :class:`veles.znicz.nn_plotting_units.Weights2D` unit.

        Arguments:
            weights_input: weights to plotting. "weights" or\
            "gradient_weights" for example
            limit: max number of pictures on one plotter
            parents: units, from whom will be link the first of\
            :class:`veles.znicz.nn_plotting_units.Weights2D` units.
        """
        name = "weights_plotter_%s" % weights_input
        prev = parents
        setattr(self, name, [])
        prev_channels = 3
        link_units = self._get_weights_source_units(weights_input)
        index = 1
        for i, layer in enumerate(self.layers):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All)):
                continue
            plt_wd = nn_plotting_units.Weights2D(
                self,
                name="%s #%s: %s" % (weights_input, index, layer["type"]),
                **self.config.weights_plotter).link_from(*prev) \
                .link_attrs(link_units[i], ("input", weights_input))
            if isinstance(self.loader, ImageLoader):
                plt_wd.link_attrs(self.loader, "color_space")
            plt_wd.input_field = "mem"
            if isinstance(self.forwards[i], conv.Conv):
                plt_wd.get_shape_from = (
                    [self.forwards[i].kx, self.forwards[i].ky, prev_channels])
                prev_channels = self.forwards[i].n_kernels
            if (layer.get("output_sample_shape") is not None and
                    layer["type"] != "softmax"):
                plt_wd.link_attrs(
                    self.forwards[i], ("get_shape_from", "input"))
                if isinstance(self.loader, ImageLoader):
                    plt_wd.link_attrs(self.loader, "color_space")
            plt_wd.gate_skip = ~self.decision.epoch_ended
            getattr(self, name).append(plt_wd)
            index += 1
            prev = plt_wd,
        return prev[0]

    def link_similar_weights_plotter(self, weights_input, *parents):
        """
        Creates the list of instances of
        :class:`veles.znicz.diversity.SimilarWeights2D` units.
        Links the first :class:`veles.znicz.diversity.SimilarWeights2D` unit
        with \*parents.
        Links each :class:`veles.znicz.diversity.SimilarWeights2D` unit from
        previous :class:`veles.znicz.diversity.SimilarWeights2D` unit.
        Links attributes of :class:`veles.znicz.diversity.SimilarWeights2D`
        units from attributes of :class:`veles.znicz.decision.DecisionBase`
        descendant, :class:`veles.loader.base.Loader` descendant,
        :class:`veles.znicz.nn_units.ForwardBase` descendant and
        :class:`veles.znicz.nn_plotting_units.Weights2D` unit.
        Returns the last of :class:`veles.znicz.diversity.SimilarWeights2D`
        units.

        Arguments:
            weights_input: weights to plotting. "weights" or\
            "gradient_weights" for example
            parents: units, from whom will be link the first of\
            :class:`veles.znicz.diversity.SimilarWeights2D` units.
        """
        self.similar_weights_plotter = []
        prev = parents
        k = 0
        n = 0
        link_units = self._get_weights_source_units(weights_input)
        for i, layer in enumerate(self.layers):
            if (not isinstance(self.forwards[i], conv.Conv) and
                    not isinstance(self.forwards[i], all2all.All2All)):
                k += 1
                n = i - k
                continue
            plt_mx = diversity.SimilarWeights2D(
                self, name="%s %s [similar]" % (i + 1, layer["type"]),
                **self.dictify(self.config.similar_weights_plotter)) \
                .link_attrs(link_units[i], ("input", weights_input)) \
                .link_from(*prev)
            plt_mx.gate_skip = ~self.decision.epoch_ended
            plt_mx.input_field = "mem"
            if isinstance(self.loader, ImageLoader):
                plt_mx.link_attrs(self.loader, "color_space")
            wd_plt = self.weights_plotter
            if n != 0:
                plt_mx.get_shape_from = wd_plt[n].get_shape_from
            if (layer.get("output_sample_shape") is not None and
                    layer["type"] != "softmax"):
                plt_mx.link_attrs(
                    self.forwards[i], ("get_shape_from", "input"))
                if isinstance(self.loader, ImageLoader):
                    plt_mx.link_attrs(self.loader, "color_space")
            self.similar_weights_plotter.append(plt_mx)
            prev = plt_mx,
        self.similar_weights_plotter[0].clear_plot = True
        self.similar_weights_plotter[-1].redraw_plot = True
        return prev[0]

    @StandardWorkflowBase.reset_unit
    @StandardWorkflowBase.check_forward_units
    @StandardWorkflowBase.check_backward_units
    def link_table_plotter(self, *parents):
        """
        Creates instance of :class:`veles.plotting_units.TableMaxMin` unit.
        Links :class:`veles.plotting_units.TableMaxMin` unit with \*parents.
        Links attributes of :class:`veles.plotting_units.TableMaxMin` with
        attributes of :class:`veles.znicz.decision.DecisionBase` descendant,
        :class:`veles.znicz.nn_units.GradientDescentBase` descendant units ,
        :class:`veles.znicz.nn_units.ForwardBase` descendant.
        Returns instance of :class:`veles.plotting_units.TableMaxMin` unit.

        Arguments:
            parents: units to link this one from.
            :class:`veles.plotting_units.TableMaxMin` unit.
        """
        self.table_plotter = plotting_units.TableMaxMin(self, name="Max, Min")
        del self.table_plotter.y[:]
        del self.table_plotter.col_labels[:]
        for i, layer in enumerate(self.layers):
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
        self.table_plotter.link_from(*parents)
        self.table_plotter.gate_skip = ~self.decision.epoch_ended
        return self.table_plotter

    def link_mse_plotter(self, *parents):
        """
        Creates the list of instances of
        :class:`veles.plotting_units.AccumulatingPlotter`.
        Links the first :class:`veles.plotting_units.AccumulatingPlotter` unit
        with \*parents.
        Links each :class:`veles.plotting_units.AccumulatingPlotter` unit from
        previous :class:`veles.plotting_units.AccumulatingPlotter` unit.
        Links attributes of :class:`veles.plotting_units.AccumulatingPlotter`
        units from attributes of :class:`veles.znicz.decision.DecisionBase`
        descendant (epoch_metrics).
        Returns the last instance of
        :class:`veles.plotting_units.AccumulatingPlotter`.

        Arguments:
            parents: units, from whom will be link the first of\
            :class:`veles.plotting_units.AccumulatingPlotter` units.
        """
        prev = parents
        self.mse_plotter = []
        styles = ["r-", "b-", "k-"]
        for i in 1, 2:
            plotter = plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]).link_from(*prev) \
                .link_attrs(self.decision, ("input", "epoch_metrics"))
            plotter.gate_skip = ~self.decision.epoch_ended
            plotter.input_field = i
            self.mse_plotter.append(plotter)
            prev = plotter,
        self.mse_plotter[0].clear_plot = True
        self.mse_plotter[-1].redraw_plot = True
        return prev[0]

    def link_min_max_plotter(self, is_min, *parents):
        """
        Creates the list of instances of
        :class:`veles.plotting_units.AccumulatingPlotter`.
        Links the first :class:`veles.plotting_units.AccumulatingPlotter` unit
        with \*parents.
        Links each :class:`veles.plotting_units.AccumulatingPlotter` unit from
        previous :class:`veles.plotting_units.AccumulatingPlotter` unit.
        Links attributes of :class:`veles.plotting_units.AccumulatingPlotter`
        units from attributes of :class:`veles.znicz.decision.DecisionBase`
        descendant (epoch_metrics).
        Returns the last instance of
        :class:`veles.plotting_units.AccumulatingPlotter`.
        Arguments:
            is_min: True if linking min plotter, otherwise, False for max.
            parents: units, from whom will be link the first of\
            :class:`veles.plotting_units.AccumulatingPlotter` units.
        """
        prev = parents
        if is_min:
            plotters = self.min_plotter = []
        else:
            plotters = self.max_plotter = []
        styles = ["", "", "k:" if is_min else "k--"]
        for i, style in enumerate(styles):
            if len(style) == 0:
                continue
            plotter = plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=style).link_from(*prev) \
                .link_attrs(self.decision, ("input", "epoch_metrics"))
            plotter.gate_skip = ~self.decision.epoch_ended
            plotter.input_field = i
            plotter.input_offset = 2 if is_min else 1
            plotters.append(plotter)
            prev = plotter,
        plotters[-1].redraw_plot = True
        return prev[0]

    @StandardWorkflowBase.reset_unit
    @StandardWorkflowBase.check_forward_units
    def link_image_plotter(self, *parents):
        """
        Creates instance of :class:`veles.plotting_units.ImagePlotter` unit.
        Links :class:`veles.plotting_units.ImagePlotter` unit with \*parents.
        Links attributes of :class:`veles.plotting_units.ImagePlotter` with
        attributes of :class:`veles.znicz.decision.DecisionBase` descendant,
        :class:`veles.znicz.nn_units.ForwardBase` descendant.
        Returns instance of :class:`veles.plotting_units.ImagePlotter` unit.

        Arguments:
            parents: units to link this one from.
            :class:`veles.plotting_units.ImagePlotter` unit.
        """
        self.image_plotter = plotting_units.ImagePlotter(
            self, name="output sample").link_from(*parents)
        self.image_plotter.inputs.append(self.forwards[-1].output)
        self.image_plotter.input_fields.append(0)
        self.image_plotter.inputs.append(self.forwards[0].input)
        self.image_plotter.input_fields.append(0)
        self.image_plotter.gate_skip = ~self.decision.epoch_ended
        return self.image_plotter

    @StandardWorkflowBase.reset_unit
    @StandardWorkflowBase.check_forward_units
    def link_immediate_plotter(self, *parents):
        """
        Creates instance of :class:`veles.plotting_units.ImmediatePlotter`
        unit.
        Links :class:`veles.plotting_units.ImmediatePlotter` unit with
        \*parents.
        Links attributes of :class:`veles.plotting_units.ImmediatePlotter`
        from attributes of :class:`veles.znicz.decision.DecisionBase`
        descendant, :class:`veles.znicz.nn_units.ForwardBase` descendant,
        :class:`veles.loader.base.Loader` descendant.
        Returns instance of :class:`veles.plotting_units.ImmediatePlotter`
        unit.

        Arguments:
            parents: units to link this one from.
            :class:`veles.plotting_units.ImmediatePlotter` unit.
        """
        self.immediate_plotter = plotting_units.ImmediatePlotter(
            self, name="ImmediatePlotter", ylim=[-1.1, 1.1]) \
            .link_from(*parents)
        self.immediate_plotter.gate_skip = ~self.decision.epoch_ended
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
        return self.immediate_plotter

    def link_result_unit(self):
        """
        Creates instance of
        :class:`veles.znicz.standard_workflow.ForwardWorkflowExtractor` unit.
        Links :class:`veles.znicz.decision.DecisionBase` descendant from
        :class:`veles.znicz.standard_workflow.ForwardWorkflowExtractor`.
        Returns instance of
        :class:`veles.znicz.standard_workflow.ForwardWorkflowExtractor`.
        """
        self.result_unit = self.ForwardWorkflowExtractor(
            self, loader_name=self.result_loader_name,
            loader_config=self.config.result_loader,
            result_unit_factory=self.result_unit_factory)
        self.decision.link_from(self.result_unit)
        self.result_unit.gate_block = ~self.decision.complete
        return self.result_unit

    @StandardWorkflowBase.reset_unit
    def link_data_saver(self, *parents):
        """
        Creates instance of :class:`veles.loader.saver.MinibatchesSaver` unit.
        Links :class:`veles.loader.saver.MinibatchesSaver` unit with
        \*parents.
        Links attributes of :class:`veles.loader.saver.MinibatchesSaver` units
        from attributes of :class:`veles.loader.base.Loader` descendant
        Returns instance of :class:`veles.loader.saver.MinibatchesSaver` unit.

        Arguments:
            parents: units to link this one from.
            :class:`veles.loader.saver.MinibatchesSaver` unit.
        """
        self.data_saver = MinibatchesSaver(
            self, **self.config.data_saver).link_from(*parents).link_attrs(
            self.loader, "shuffle_limit", "minibatch_class", "minibatch_data",
            "minibatch_labels", "class_lengths", "max_minibatch_size",
            "has_labels", "minibatch_size")
        return self.data_saver

    def _check_forwards(self):
        if len(self.forwards) == 0:
            raise ValueError(
                "Please create forwards in workflow first. "
                "You can use link_forwards() function")

    def _check_gds(self):
        if len(self.gds) == 0:
            raise ValueError(
                "Please create gds in workflow first. "
                "For that you can use link_gds() function")

    def _get_weights_source_units(self, weights_input):
        if weights_input == "weights":
            self._check_forwards()
            return self.forwards
        elif weights_input == "gradient_weights":
            self._check_gds()
            return self.gds
        raise ValueError(
            "weights_input should be 'weights' or 'gradient_weights'")


@implementer(IUnit, IDistributable)
class ForwardWorkflowExtractor(Unit, TriviallyDistributable):
    """
    Class to extract core of Neural Network without back propagation.
    """

    def __init__(self, workflow, **kwargs):
        assert isinstance(workflow, StandardWorkflow)
        super(ForwardWorkflowExtractor, self).__init__(workflow, **kwargs)
        self.loader_name = kwargs["loader_name"]
        self.loader_config = kwargs["loader_config"]
        self.result_unit_factory = kwargs["result_unit_factory"]
        self.cyclic = kwargs.get("cyclic", False)
        self.forward_workflow = False  # StandardWorkflow, enable mutable links

    def initialize(self, **kwargs):
        pass

    def run(self):
        self.forward_workflow = self.workflow.extract_forward_workflow(
            loader_name=self.loader_name, loader_config=self.loader_config,
            result_unit_factory=self.result_unit_factory, cyclic=self.cyclic)

    def apply_data_from_slave(self, data, slave):
        if not self.gate_block:
            self.run()
