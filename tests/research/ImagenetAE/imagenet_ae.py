# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on July 4, 2014

Model created for object recognition. Dataset - Imagenet.
Model - convolutional neural network, dynamically
constructed, with pretraining of all layers one by one with autoencoder.

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


import os

import numpy
from zope.interface import implementer

from veles.config import root
import veles.error as error
from veles.mutable import Bool
import veles.znicz.conv as conv
import veles.znicz.evaluator as evaluator
from veles.normalization import NoneNormalizer
import veles.znicz.deconv as deconv
import veles.znicz.gd_deconv as gd_deconv
import veles.znicz.image_saver as image_saver
from veles.znicz.loader.imagenet_loader import ImagenetLoaderBase
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.pooling as pooling
import veles.znicz.depooling as depooling
import veles.znicz.activation as activation
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.gd_conv as gd_conv
import veles.znicz.gd as gd
from veles.znicz.standard_workflow import StandardWorkflow
from veles.units import IUnit, Unit
from veles.distributable import IDistributable
import veles.prng as prng
from veles.prng.uniform import Uniform
from veles.dummy import DummyWorkflow


@implementer(IUnit, IDistributable)
class Destroyer(Unit):
    """
    Modification of EndPoint Unit, created for Convolutional Autoencoder
    """
    def initialize(self, **kwargs):
        pass

    def run(self):
        if not self.is_slave:
            self.workflow.on_workflow_finished()

    def generate_data_for_master(self):
        return True

    def generate_data_for_slave(self, slave):
        return None

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        if not bool(self.gate_block) and not bool(self.gate_skip):
            self.run()

    def drop_slave(self, slave):
        pass


class ImagenetAELoader(ImagenetLoaderBase):
    MAPPING = "imagenet_ae_loader"

    def __init__(self, workflow, **kwargs):
        super(ImagenetAELoader, self).__init__(workflow, **kwargs)
        self.target_normalizer = NoneNormalizer

    def load_data(self):
        super(ImagenetAELoader, self).load_data()
        if (self.matrixes_filename is None or
                not os.path.exists(self.matrixes_filename)):
            raise OSError(
                "matrixes_filename %s does not exist or None. Please "
                "specify path to file with mean and disp matrixes. If you "
                "don't have pickle file with mean and disp matrixes, generate"
                " it with preparation_imagenet.py" % self.matrixes_filename)
        self.load_mean()

    def create_minibatch_data(self):
        sh = [self.max_minibatch_size]
        sh.extend((self.final_sy, self.final_sx, self.channels))
        self.minibatch_data.mem = numpy.zeros(sh, dtype=numpy.uint8)


class ImagenetAEWorkflow(StandardWorkflow):
    """
    Workflow for recognition of images on base of Imagenet data.
    Model - Concolutional Neural Network with pretraining of each layer with
    autoencoder. New layers will be added to the model if
    workflow is loaded from snapshot and option from_snapshot_add_layer
    is True. When all layers are trained separately, workflow switches to the
    fine tuning stage and model is training with all layers.
    """
    def __init__(self, workflow, **kwargs):
        self.from_snapshot_add_layer = kwargs["from_snapshot_add_layer"]
        self.add_epochs = kwargs["add_epochs"]
        self.fine_tuning_noise = kwargs["fine_tuning_noise"]
        self.decision_mse_config = kwargs["decision_mse_config"]
        self.decision_gd_config = kwargs["decision_gd_config"]
        self.layer_map = nn_units.MatchingObject.mapping
        super(ImagenetAEWorkflow, self).__init__(workflow, **kwargs)

    def init_unpickled(self):
        super(ImagenetAEWorkflow, self).init_unpickled()
        self.last_de_map = {  # If the last unit in ae-layer => replace with:
            pooling.StochasticAbsPooling:
            pooling.StochasticAbsPoolingDepooling,
            pooling.StochasticPooling: pooling.StochasticPoolingDepooling}
        self.last_de_unmap = {  # When adding next autoencoder => replace with:
            pooling.StochasticAbsPoolingDepooling:
            pooling.StochasticAbsPooling,
            pooling.StochasticPoolingDepooling: pooling.StochasticPooling}
        self.de_map = {
            conv.Conv: deconv.Deconv,
            pooling.StochasticAbsPooling: depooling.Depooling,
            pooling.MaxAbsPooling: depooling.Depooling,
            pooling.StochasticPooling: depooling.Depooling,
            pooling.MaxPooling: depooling.Depooling}
        self.gd_map = {}
        for layer_type, forw_back in (dict(self.layer_map)).items():
            if len(forw_back) > 1:
                if issubclass(forw_back[0], nn_units.ForwardBase):
                    self.gd_map[forw_back[0]] = forw_back[1]
                if issubclass(forw_back[1], nn_units.ForwardBase):
                    self.gd_map[forw_back[1]] = forw_back[0]

    def initialize(self, device, **kwargs):
        if (self.forwards[0].weights.mem is not None and
                self.from_snapshot_add_layer):
            self.info("Restoring from snapshot detected, "
                      "will adjust the workflow")
            self.adjust_workflow()
            self.info("Workflow adjusted, will initialize now")
        else:
            self.decision.max_epochs += self.add_epochs
        self.decision.complete <<= False
        self.info("Set decision.max_epochs to %d and complete=False",
                  self.decision.max_epochs)
        super(ImagenetAEWorkflow, self).initialize(device, **kwargs)
        self.dump_shapes()

    def create_workflow(self):
        self.link_repeater(self.start_point)
        self.link_loader(self.repeater)

        prev = self.link_meandispnorm(self.loader)

        self.number_ae_blocks = 0
        ae_units, ae_layers, last_conv, _, prev = self._add_forwards(
            prev, self.layers, "set_pool")
        self._add_deconv_units(ae_units, ae_layers, prev)
        self.fix_dropout()
        self.link_evaluator_mse(self.forwards[-1])
        self.loss_function = "mse"
        self.decision_name = "decision_mse"
        self.apply_config(decision_config=self.decision_mse_config)
        self.link_decision(self.evaluator)
        self.decision.autoencoder = True
        self.link_snapshotter(self.decision)
        self.link_rollback(self.snapshotter)

        self.add_gds_and_plots(ae_layers, last_conv, ae_units)

        self.destroyer = Destroyer(self)
        self.link_end_point(self.gds[0])

    def adjust_workflow(self):
        self.info("Will extend %d autoencoder layers", self.number_ae_blocks)

        layers = self.layers
        n_ae = 0
        i_layer = 0
        i_fwd = 0
        i_fwd_last = 0
        for layer in layers:
            i_layer += 1
            if layer["type"] == "ae_begin":
                continue
            if layer["type"] == "ae_end":
                i_fwd_last = i_fwd
                n_ae += 1
                if n_ae >= self.number_ae_blocks:
                    break
                continue
            i_fwd += 1
        else:
            self.warning("Will switch to the fine-tuning task")
            return self.switch_to_fine_tuning()

        # remove all forwards after the last autoencoder block
        i_fwd = i_fwd_last
        for i in range(i_fwd, len(self.forwards)):
            self.forwards[i].unlink_all()
            self.del_ref(self.forwards[i])
        del self.forwards[i_fwd:]
        last_fwd = self.forwards[-1]
        prev = last_fwd

        if prev.__class__ in self.last_de_unmap:
            self.info("Replacing pooling-depooling with pooling")
            layer = prev.layer
            __, kwargs, _ = self._get_layer_type_kwargs(layer)
            uniform = prev.uniform
            prev.unlink_all()
            self.del_ref(prev)
            unit = self.last_de_unmap[prev.__class__](self, **kwargs)
            unit.layer = layer
            self.forwards[-1] = unit
            unit.uniform = uniform
            unit.link_from(self.forwards[-2])
            unit.link_attrs(self.forwards[-2], ("input", "output"))
            self.create_output(unit)
            prev = unit

        ae_units, ae_layers, last_conv, in_ae, prev = self._add_forwards(
            prev, layers[i_layer:], "get_pool")

        if in_ae:
            self._add_deconv_units(ae_units, ae_layers, prev)

            unit = self.evaluator
            unit.link_from(self.forwards[-1])
            unit.link_attrs(self.forwards[-1], "output")
            unit.link_attrs(last_conv, ("target", "input"))

            assert len(self.gds) == 1

            self.gds[0].unlink_all()
            self.del_ref(self.gds[0])
            del self.gds[:]

            self.rollback.reset()
            self.add_gds_and_plots(ae_layers, last_conv, ae_units)

            self.reset_best_error()
            self.decision.max_epochs += \
                root.imagenet_ae.decision_mse.max_epochs
            last = self.gds[0]

        else:
            self.info("No more autoencoder levels, "
                      "will switch to the classification task")
            self.number_ae_blocks += 1
            self.fix_dropout()
            self.link_image_saver(self.forwards[-1])
            self.evaluator_name = "evaluator_softmax"
            self.unlink_unit("evaluator")
            self.link_evaluator(self.image_saver)
            self.apply_config(decision_config=self.decision_gd_config)
            self.loss_function = "softmax"
            self.decision_name = "decision_gd"
            self.link_decision(self.evaluator)
            self.link_snapshotter(self.decision)
            self.link_rollback(self.snapshotter)

            self.image_saver.gate_skip = ~self.decision.improved
            self.image_saver.link_attrs(self.snapshotter,
                                        ("this_save_time", "time"))

            self.rollback.gate_skip = (~self.loader.epoch_ended |
                                       self.decision.complete)
            self.rollback.improved = self.decision.train_improved

            assert len(self.gds) == 1

            for gd_ in self.gds:
                gd_.unlink_all()
                self.del_ref(gd_)
            del self.gds[:]

            self.rollback.reset()

            prev = self.rollback
            prev_gd = self.evaluator
            gds = []

            for layer in self.layers:
                if (layer["type"].find("ae_begin") >= 0 or
                        layer["type"].find("ae_end") >= 0):
                    self.layers.remove(layer)

            for i in range(len(self.forwards) - 1, i_fwd - 1, -1):
                __, kwargs, _ = self._get_layer_type_kwargs(
                    self.forwards[i].layer)
                unit = self.gd_map[self.forwards[i].__class__](self, **kwargs)
                gds.append(unit)
                if prev is not None:
                    unit.link_from(prev)
                if isinstance(prev_gd, evaluator.EvaluatorBase):
                    unit.link_attrs(prev_gd, "err_output")
                else:
                    unit.link_attrs(prev_gd, ("err_output", "err_input"))
                unit.link_attrs(self.forwards[i], "weights", "input", "output")
                if hasattr(self.forwards[i], "input_offset"):
                    unit.link_attrs(self.forwards[i], "input_offset")
                if hasattr(self.forwards[i], "mask"):
                    unit.link_attrs(self.forwards[i], "mask")
                if self.forwards[i].bias is not None:
                    unit.link_attrs(self.forwards[i], "bias")
                unit.gate_skip = self.decision.gd_skip
                prev_gd = unit
                prev = unit

            for gd_ in self.gds:
                if not isinstance(gd_, activation.Activation):
                    self.rollback.add_gd(gd_)

            # Strip gd's without weights
            for i in range(len(gds) - 1, -1, -1):
                if (isinstance(gds[i], gd.GradientDescent) or
                        isinstance(gds[i], gd_conv.GradientDescentConv)):
                    break
                unit = gds.pop(-1)
                unit.unlink_all()
                self.del_ref(unit)
            for _ in gds:
                self.gds.append(None)
            for i, _gd in enumerate(gds):
                self.gds[-(i + 1)] = _gd
            del gds

            self.gds[0].need_err_input = False

            prev = self.gds[0]

            if self.is_standalone:
                for unit in self.mse_plotter:
                    unit.unlink_all()
                    self.del_ref(unit)
                del self.mse_plotter[:]
                for weights_input in ("weights", "input", "output"):
                    name = "ae_weights_plotter_%s" % weights_input
                    self.unlink_unit(name)
                self.unlink_unit("plt_deconv")
                prev = self.link_error_plotter(self.gds[0])
                prev = self.link_weights_plotter("weights", prev)

            self.link_lr_adjuster(prev)

            last = self.lr_adjuster

            self.repeater.link_from(last)

        self.link_end_point(last)

    def switch_to_fine_tuning(self):
        if len(self.gds) == len(self.forwards):
            self.info("Already at fine-tune stage, continue training")
            return
        # Add gradient descent units for the remaining forward units
        self.gds[0].unlink_after()
        self.gds[0].need_err_input = True
        prev = self.gds[0]

        for i in range(len(self.forwards) - len(self.gds) - 1, -1, -1):
            if hasattr(self.forwards[i], "layer"):
                __, kwargs, _ = self._get_layer_type_kwargs(
                    self.forwards[i].layer)
            if "learning_rate_ft" in kwargs:
                kwargs["learning_rate"] = kwargs["learning_rate_ft"]
            if "learning_rate_ft_bias" in kwargs:
                kwargs["learning_rate_bias"] = kwargs["learning_rate_ft_bias"]

            gd_unit = self.gd_map[self.forwards[i].__class__](self, **kwargs)
            self.gds.insert(0, gd_unit)
            gd_unit.link_from(prev)
            if isinstance(gd_unit, gd_pooling.GDPooling):
                gd_unit.link_attrs(self.forwards[i], "kx", "ky", "sliding")
            if isinstance(gd_unit, gd_conv.GradientDescentConv):
                gd_unit.link_attrs(
                    self.forwards[i], "n_kernels", "kx", "ky", "sliding",
                    "padding", "unpack_size")
            gd_unit.link_attrs(prev, ("err_output", "err_input"))
            gd_unit.link_attrs(self.forwards[i], "weights", "input", "output")
            if hasattr(self.forwards[i], "input_offset"):
                gd_unit.link_attrs(self.forwards[i], "input_offset")
            if hasattr(self.forwards[i], "mask"):
                gd_unit.link_attrs(self.forwards[i], "mask")
            if self.forwards[i].bias is not None:
                gd_unit.link_attrs(self.forwards[i], "bias")
                gd_unit.gate_skip = self.decision.gd_skip
            prev = gd_unit

        self.gds[0].need_err_input = False

        prev = self.gds[0]

        if self.is_standalone:
            for unit in self.mse_plotter:
                unit.unlink_all()
                self.del_ref(unit)
            del self.mse_plotter[:]
            for weights_input in ("weights", "input", "output"):
                name = "ae_weights_plotter_%s" % weights_input
                self.unlink_unit(name)
            prev = self.link_error_plotter(self.gds[0])
            prev = self.link_weights_plotter("weights", prev)

        self.link_lr_adjuster(prev)
        self.repeater.link_from(self.lr_adjuster)

        self.rollback.reset()
        noise = float(self.fine_tuning_noise)
        for unit in self.gds:
            if not isinstance(unit, activation.Activation):
                self.rollback.add_gd(unit)
            if not noise:
                continue
            if unit.weights:
                weights = unit.weights.plain
                weights += prng.get().normal(0, noise, unit.weights.size)
            if unit.bias:
                bias = unit.bias.plain
                bias += prng.get().normal(0, noise, unit.bias.size)

        self.reset_best_error()
        self.decision.max_epochs += root.imagenet_ae.decision.add_epochs
        self.link_end_point(self.gds[0])

    def link_evaluator_mse(self, *parents):
        if hasattr(self, "evaluator"):
            self.evaluator.unlink_all()
            self.del_ref(self.evaluator)
        self.evaluator = evaluator.EvaluatorMSE(self)
        self.evaluator.link_from(*parents)
        self.evaluator.link_attrs(self.forwards[-1], "output")
        self.evaluator.link_attrs(
            self.loader, ("batch_size", "minibatch_size"),
            ("normalizer", "target_normalizer"))
        self.evaluator.link_attrs(self.meandispnorm, ("target", "output"))
        return self.evaluator

    def link_end_point(self, *parents):
        self.rollback.gate_block = Bool(False)
        self.rollback.gate_skip = (~self.loader.epoch_ended |
                                   self.decision.complete)
        self.end_point.unlink_all()
        self.end_point.link_from(*parents)
        self.end_point.gate_block = ~self.decision.complete
        self.loader.gate_block = self.decision.complete
        if not hasattr(self, "destroyer"):
            self.destroyer = Destroyer(self)
        self.destroyer.unlink_all()
        self.destroyer.link_from(*parents)
        self.destroyer.gate_block = ~self.decision.complete

    def _add_forwards(self, prev, layers, pool):
        """
        Add one block at the time to train with autoencoder: conv -> pooling
        """
        ae_units = []
        ae_layers = []
        last_conv = None
        in_ae = False
        self.uniform = None
        for layer in layers:
            if layer["type"] == "ae_begin":
                self.info("Autoencoder block begin")
                in_ae = True
                continue
            if layer["type"] == "ae_end":
                self.info("Autoencoder block end")
                self.info("One AE at a time, so skipping other layers")
                self.number_ae_blocks += 1
                break
            if layer["type"].find("conv") > -1:
                if in_ae:
                    layer["include_bias"] = False
                layer["->"]["padding"] = deconv.Deconv.compute_padding(
                    self.loader.sx, self.loader.sy,
                    layer["->"]["kx"], layer["->"]["ky"],
                    layer["->"]["sliding"])
            tpe, kwargs, _ = self._get_layer_type_kwargs(layer)
            forward_unit = self.layer_map[tpe].forward(self, **kwargs)
            forward_unit.layer = dict(layer)
            if in_ae:
                ae_units.append(forward_unit)
                ae_layers.append(layer)
            getattr(self, pool)(forward_unit)
            self.forwards.append(forward_unit)
            forward_unit.link_from(prev)
            if hasattr(prev, "output"):
                forward_unit.link_attrs(prev, ("input", "output"))
            prev = forward_unit
            if layer["type"].find("conv") > -1 and in_ae:
                last_conv = prev

        if last_conv is None and in_ae:
            raise error.BadFormatError("No convolutional layer found")

        return ae_units, ae_layers, last_conv, in_ae, prev

    def get_pool(self, forward_unit):
        if isinstance(forward_unit, pooling.StochasticPoolingBase):
            forward_unit.uniform = self.uniform

    def set_pool(self, forward_unit):
        if isinstance(forward_unit, pooling.StochasticPoolingBase):
            if self.uniform is None:
                self.uniform = Uniform(DummyWorkflow(), num_states=512)
                forward_unit.uniform = self.uniform

    def _add_deconv_units(self, ae_units, ae_layers, prev):
        """
        Add deconvolutional and depooling units
        """
        deconv_units = []
        for i in range(len(ae_units) - 1, -1, -1):
            __, kwargs, _ = self._get_layer_type_kwargs(ae_layers[i])
            if (i == len(ae_units) - 1 and
                    id(ae_units[-1]) == id(self.forwards[-1]) and
                    ae_units[-1].__class__ in self.last_de_map):
                self.info("Replacing pooling with pooling-depooling")
                depool_unit = self.last_de_map[ae_units[-1].__class__](
                    self, **kwargs)
                depool_unit.uniform = ae_units[-1].uniform
                depool_unit.layer = ae_units[-1].layer
                depool_unit.link_attrs(self.forwards[-2], ("input", "output"))

                del self.forwards[-1]
                ae_units[-1].unlink_all()
                self.del_ref(ae_units[-1])

                ae_units[-1] = depool_unit
                self.forwards.append(depool_unit)
                depool_unit.link_from(self.forwards[-2])
                prev = depool_unit
                # for the later assert to work:
                deconv_units.append(depool_unit)
                continue
            de_unit = self.de_map[ae_units[i].__class__](self, **kwargs)
            if isinstance(de_unit, deconv.Deconv):
                de_unit.link_conv_attrs(ae_units[i])
                self.deconv = de_unit
                de_unit.unsafe_padding = True
                deconv_units.append(de_unit)
            self.forwards.append(de_unit)
            de_unit.link_from(prev)
            for dst_src in (("weights", "weights"),
                            ("output_shape_source", "input"),
                            ("output_offset", "input_offset")):
                if hasattr(de_unit, dst_src[0]):
                    de_unit.link_attrs(ae_units[i], dst_src)
            if isinstance(prev, pooling.StochasticPoolingDepooling):
                de_unit.link_attrs(prev, "input")
            else:
                de_unit.link_attrs(prev, ("input", "output"))
            prev = de_unit

        assert len(ae_units) == len(deconv_units)

    def add_gds_and_plots(self, ae_layers, last_conv, ae_units):
        __, _, kwargs = self._get_layer_type_kwargs(ae_layers[0])
        gd_deconv_unit = self.gd_map[self.forwards[-1].__class__](
            self, **kwargs)
        if isinstance(gd_deconv_unit, gd_deconv.GDDeconv):
            gd_deconv_unit.link_attrs(
                self.deconv, "weights", "input", "hits", "n_kernels",
                "kx", "ky", "sliding", "padding", "unpack_size")
        self.gds.append(gd_deconv_unit)
        gd_deconv_unit.link_attrs(self.evaluator, "err_output")
        gd_deconv_unit.link_attrs(self.forwards[-1], "weights", "input")
        gd_deconv_unit.gate_skip = self.decision.gd_skip
        self.rollback.add_gd(gd_deconv_unit)

        assert len(self.gds) == 1  # unit must be GDDeconv

        self.gds[0].need_err_input = False
        self.repeater.link_from(self.gds[0])

        prev = self.add_ae_plotters(self.rollback, last_conv, ae_units[-1])
        self.gds[-1].link_from(prev)

    def link_ae_weights_plotter(
            self, weights_input, last_conv, last_ae, *parents):
        name = "ae_weights_plotter_%s" % weights_input
        self.unlink_unit(name)
        setattr(self, name, nn_plotting_units.Weights2D(
            self, name=weights_input, **self.config.weights_plotter))
        if weights_input == "weights":
            getattr(self, name).get_shape_from = [
                last_conv.kx, last_conv.ky, last_conv.input]
        if weights_input == "output":
            if isinstance(last_ae, pooling.StochasticPoolingDepooling):
                getattr(self, name).link_attrs(last_ae, "input")
            else:
                getattr(self, name).link_attrs(last_ae, ("input", "output"))
        else:
            getattr(self, name).link_attrs(last_conv, ("input", weights_input))
        getattr(self, name).link_from(*parents)
        getattr(self, name).gate_skip = ~self.decision.epoch_ended
        return getattr(self, name)

    def add_ae_plotters(self, prev, last_conv, last_ae):
        if not self.is_standalone:
            return prev
        if hasattr(self, "mse_plotter"):
            for unit in self.mse_plotter:
                unit.unlink_all()
                self.del_ref(unit)
            del self.mse_plotter[:]
        last_mse = self.link_mse_plotter(prev)
        last_weights = self.link_ae_weights_plotter(
            "weights", last_conv, last_ae, last_mse)
        last_input = self.link_ae_weights_plotter(
            "input", last_conv, last_ae, last_weights)
        last_output = self.link_ae_weights_plotter(
            "output", last_conv, last_ae, last_input)

        self.unlink_unit("plt_deconv")
        self.plt_deconv = nn_plotting_units.Weights2D(
            self, name="Deconv result", limit=256, split_channels=False)
        self.plt_deconv.link_attrs(self.forwards[-1], ("input", "output"))
        self.plt_deconv.link_from(last_output)
        self.plt_deconv.gate_skip = ~self.decision.epoch_ended

        return self.plt_deconv

    def dump_shapes(self):
        self.info("Input-Output Shapes:")
        for i, fwd in enumerate(self.forwards):
            self.info("%d: %s: %s => %s", i, repr(fwd),
                      str(fwd.input.shape) if fwd.input else "None",
                      str(fwd.output.shape) if fwd.output else "None")

    def reset_best_error(self):
        """
        Reset last best error for modified workflow
        """
        self.decision.min_validation_mse = 1.0e30
        self.decision.min_train_validation_mse = 1.0e30
        self.decision.min_train_mse = 1.0e30
        self.decision.min_validation_n_err = 1.0e30
        self.decision.min_train_validation_n_err = 1.0e30
        self.decision.min_train_n_err = 1.0e30

    def reset_weights(self):
        for unit in self:
            if not hasattr(unit, "layer"):
                continue
            if hasattr(unit, "weights") and unit.weights:
                self.info("RESETTING weights for %s", repr(unit))
                weights = unit.weights.plain
                weights *= 0
                weights += prng.get().normal(
                    0, unit.layer["->"]["weights_stddev"], unit.weights.size)
            if hasattr(unit, "bias") and unit.bias:
                self.info("RESETTING bias for %s", repr(unit))
                bias = unit.bias.plain
                bias *= 0
                bias += prng.get().normal(
                    0, unit.layer["->"]["bias_stddev"], unit.bias.size)

    def create_output(self, unit):
        """
        Create output for polling
        """
        unit._batch_size = unit.input.shape[0]
        unit._sy = unit.input.shape[1]
        unit._sx = unit.input.shape[2]
        unit._n_channels = unit.input.size // (unit._batch_size *
                                               unit._sx * unit._sy)

        last_x = unit._sx - unit.kx
        last_y = unit._sy - unit.ky
        if last_x % unit.sliding[0] == 0:
            unit._out_sx = last_x // unit.sliding[0] + 1
        else:
            unit._out_sx = last_x // unit.sliding[0] + 2
        if last_y % unit.sliding[1] == 0:
            unit._out_sy = last_y // unit.sliding[1] + 1
        else:
            unit._out_sy = last_y // unit.sliding[1] + 2
            unit._output_size = (
                unit._n_channels * unit._out_sx *
                unit._out_sy * unit._batch_size)
            unit._output_shape = (
                unit._batch_size, unit._out_sy, unit._out_sx, unit._n_channels)
        if (not unit._no_output and
                (not unit.output or unit.output.size != unit._output_size)):
            unit.output.reset()
            unit.output.mem = numpy.zeros(
                unit.output_shape, dtype=unit.input.dtype)

    def link_image_saver(self, *parents):
        self.image_saver = image_saver.ImageSaver(
            self, **self.config.image_saver)
        self.image_saver.link_from(*parents)
        self.image_saver.link_attrs(self.forwards[-1], "output", "max_idx")
        self.image_saver.link_attrs(
            self.loader,
            ("indices", "minibatch_indices"),
            ("labels", "minibatch_labels"),
            "minibatch_class", "minibatch_size")
        self.image_saver.link_attrs(self.meandispnorm, ("input", "output"))
        return self.image_saver


def run(load, main):
    load(ImagenetAEWorkflow,
         loss_function="mse",
         layers=root.imagenet_ae.layers,
         loader_name=root.imagenet_ae.loader_name,
         from_snapshot_add_layer=root.imagenet_ae.from_snapshot_add_layer,
         weights_plotter_config=root.imagenet_ae.weights_plotter,
         lr_adjuster_config=root.imagenet_ae.lr_adjuster,
         loader_config=root.imagenet_ae.loader,
         decision_mse_config=root.imagenet_ae.decision_mse,
         decision_gd_config=root.imagenet_ae.decision_gd,
         add_epochs=root.imagenet_ae.decision.add_epochs,
         fine_tuning_noise=root.imagenet_ae.fine_tuning_noise,
         image_saver_config=root.imagenet_ae.image_saver,
         rollback_config=root.imagenet_ae.rollback,
         snapshotter_config=root.imagenet_ae.snapshotter)
    main()
