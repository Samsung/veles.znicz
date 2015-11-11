# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on August 25, 2015

Standard workflow base class definition.

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

from veles.compat import from_none
import veles.error as error
from veles.plumbing import FireStarter
# Important: do not remove unused imports! It will prevent MatchingObject
# metaclass from adding the mapping in the corresponding modules
from veles.znicz import activation  # pylint: disable=W0611
from veles.znicz.all2all import All2AllSoftmax
from veles.znicz import dropout  # pylint: disable=W0611
from veles.znicz import nn_units
from veles.znicz import normalization  # pylint: disable=W0611
from veles.znicz import weights_zerofilling
from veles.loader.base import UserLoaderRegistry, LoaderMSEMixin


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

    def fix_dropout(self):
        # TODO: This is temporary fix. Need to remove it after fixing Dropout
        # TODO: for real
        followers = list(self.loader.links_to.keys())
        self.loader.unlink_after()
        self.dropout_fixer = dropout.DropoutFixer(self).link_from(self.loader)
        for u in followers:
            u().link_from(self.dropout_fixer)

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
                        numpy.prod(last_fwd.output_sample_shape) !=
                        numpy.prod(self.real_loader.targets_shape)):
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
