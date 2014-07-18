"""
Created on July 4, 2014

Imagenet recognition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import json
import numpy
import os
import pickle
from zope.interface import implementer

from veles.config import root
import veles.error as error
from veles.formats import Vector
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.znicz.conv as conv
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.loader as loader
import veles.znicz.deconv as deconv
import veles.znicz.gd_deconv as gd_deconv
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.pooling as pooling
import veles.znicz.depooling as depooling
import veles.znicz.activation as activation
import veles.znicz.all2all as all2all
import veles.znicz.gd as gd
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.gd_conv as gd_conv
from veles.znicz.nn_units import NNSnapshotter
from veles.znicz.standard_workflow import StandardWorkflow
from veles.mean_disp_normalizer import MeanDispNormalizer
from veles.units import IUnit, Unit
from veles.distributable import TriviallyDistributable
import veles.random_generator as prng
import veles.external.prettytable as prettytable

IMAGENET_BASE_PATH = os.path.join(root.common.test_dataset_root,
                                  "imagenet")
root.common.snapshot_dir = os.path.join(root.common.test_dataset_root,
                                        "imagenet/snapshots")

LR = 0.00001
WD = 0.004
GM = 0.9
L1_VS_L2 = 0.0

LRFT = 0.001
LRFTB = LRFT

LRAA = 0.01
LRBAA = LRAA * 2
WDAA = 0.004
WDBAA = WDAA
GMAA = 0.9
GMBAA = GM

FILLING = "gaussian"
STDDEV_CONV = 0.01
STDDEV_AA = 0.001

root.defaults = {
    "decision": {"fail_iterations": 25,
                 "max_epochs": 75,
                 "use_dynamic_alpha": False,
                 "do_export_weights": True},
    "snapshotter": {"prefix": "imagenet_ae"},
    "loader": {"year": "temp",
               "series": "img",
               "minibatch_size": 14},
    "imagenet": {"from_snapshot_add_layer": True,
                 "fine_tuning_noise": 1.0e-6,
                 "layers":
                 [{"type": "ae_begin"},  # 256
                  {"type": "conv", "n_kernels": 16,
                   "kx": 8, "ky": 8, "sliding": (1, 1),
                   "learning_rate": LR,
                   "learning_rate_ft": LRFT,
                   "weights_decay": WD,
                   "gradient_moment": GM,
                   "weights_filling": FILLING,
                   "weights_stddev": STDDEV_CONV,
                   "l1_vs_l2": L1_VS_L2},
                  {"type": "stochastic_abs_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                  {"type": "ae_end"},

                  {"type": "activation_mul"},
                  {"type": "ae_begin"},  # 128
                  {"type": "conv", "n_kernels": 16,
                   "kx": 8, "ky": 8, "sliding": (1, 1),
                   "learning_rate": LR,
                   "learning_rate_ft": LRFT,
                   "weights_decay": WD,
                   "gradient_moment": GM,
                   "weights_filling": FILLING,
                   "weights_stddev": STDDEV_CONV,
                   "l1_vs_l2": L1_VS_L2},
                  {"type": "stochastic_abs_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                  {"type": "ae_end"},

                  {"type": "activation_mul"},
                  {"type": "ae_begin"},  # 64
                  {"type": "conv", "n_kernels": 16,
                   "kx": 8, "ky": 8, "sliding": (1, 1),
                   "learning_rate": LR,
                   "learning_rate_ft": LRFT,
                   "weights_decay": WD,
                   "gradient_moment": GM,
                   "weights_filling": FILLING,
                   "weights_stddev": STDDEV_CONV,
                   "l1_vs_l2": L1_VS_L2},
                  {"type": "stochastic_abs_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                  {"type": "ae_end"},

                  {"type": "activation_mul"},
                  {"type": "all2all_tanh", "output_shape": 100,
                   "learning_rate": LRAA, "learning_rate_bias": LRBAA,
                   "learning_rate_ft": LRFT, "learning_rate_ft_bias": LRFTB,
                   "weights_decay": WDAA, "weights_decay_bias": WDBAA,
                   "gradient_moment": GMAA, "gradient_moment_bias": GMBAA,
                   "weights_filling": "gaussian", "bias_filling": "gaussian",
                   "weights_stddev": STDDEV_AA, "bias_stddev": STDDEV_AA,
                   "l1_vs_l2": L1_VS_L2},

                  {"type": "softmax", "output_shape": 5,
                   "learning_rate": LRAA, "learning_rate_bias": LRBAA,
                   "learning_rate_ft": LRFT, "learning_rate_ft_bias": LRFTB,
                   "weights_decay": WDAA, "weights_decay_bias": WDBAA,
                   "gradient_moment": GMAA, "gradient_moment_bias": GMBAA,
                   "weights_filling": "gaussian", "bias_filling": "gaussian",
                   "l1_vs_l2": L1_VS_L2}]}}


@implementer(IUnit)
class NNRollback(Unit, TriviallyDistributable):
    def __init__(self, workflow, **kwargs):
        super(NNRollback, self).__init__(workflow, **kwargs)
        self.lr_plus = kwargs.get("lr_plus", 1.1)
        self.lr_minus = kwargs.get("lr_minus", 0.55)
        self.plus_steps = kwargs.get("plus_steps", 1)
        self.minus_steps = kwargs.get("minus_steps", 4)
        self._plus_steps = self.plus_steps
        self._minus_steps = self.minus_steps
        self.improved = None
        self.demand("improved")
        self._gds = {}

    def initialize(self, **kwargs):
        pass

    def run(self):
        if self.improved:
            self._plus_steps += 1
            if self._plus_steps < self.plus_steps:
                return
            self._plus_steps = 0
            self._minus_steps = 0
            for _gd, kv in self._gds.items():
                k = kv["lr_plus"]
                if k is None:
                    k = self.lr_plus
                _gd.learning_rate *= k
                _gd.learning_rate_bias *= k
                self.info("Increased lr of %s by %.2f, new_lr %.2e",
                          repr(_gd), k, _gd.learning_rate)
                if _gd.weights:
                    _gd.weights.map_read()
                    kv["weights"] = _gd.weights.mem.copy()
                if _gd.bias:
                    _gd.bias.map_read()
                    kv["bias"] = _gd.bias.mem.copy()
        else:
            self._minus_steps += 1
            if self._minus_steps < self.minus_steps:
                return
            self._minus_steps = 0
            self._plus_steps = 0
            for _gd, kv in self._gds.items():
                k = kv["lr_minus"]
                if k is None:
                    k = self.lr_minus
                _gd.learning_rate *= k
                _gd.learning_rate_bias *= k
                self.info("Decreased lr of %s by %.2f, new_lr %.2e",
                          repr(_gd), k, _gd.learning_rate)
                if _gd.weights:
                    if kv["weights"] is None:
                        self.warning("No rollback for weights")
                    else:
                        _gd.weights.map_invalidate()
                        _gd.weights.mem[:] = kv["weights"]
                if _gd.bias:
                    if kv["bias"] is None:
                        self.warning("No rollback for bias")
                    else:
                        _gd.bias.map_invalidate()
                        _gd.bias.mem[:] = kv["bias"]

    def reset(self):
        self._gds.clear()

    def add_gd(self, _gd, lr_plus=None, lr_minus=None):
        kv = self._gds.get(_gd, {})
        kv["lr_plus"] = lr_plus
        kv["lr_minus"] = lr_minus
        kv["weights"] = None
        kv["bias"] = None
        self._gds[_gd] = kv


@implementer(loader.ILoader)
class Loader(loader.Loader):
    """loads imagenet from samples.dat, labels.pickle"""
    def __init__(self, workflow, **kwargs):
        super(Loader, self).__init__(workflow, **kwargs)
        self.mean = Vector()
        self.rdisp = Vector()
        self.file_samples = ""
        self.sx = 288
        self.sy = 288

    def init_unpickled(self):
        super(Loader, self).init_unpickled()
        self.original_labels = None

    def __getstate__(self):
        stt = super(Loader, self).__getstate__()
        stt["original_labels"] = None
        stt["file_samples"] = None
        return stt

    def load_data(self):
        self.original_labels = []

        with open(root.loader.names_labels_filename, "rb") as fin:
            for lbl in pickle.load(fin):
                self.original_labels.append(int(lbl))
        self.info("Labels (min max): %d %d",
                  numpy.min(self.original_labels),
                  numpy.max(self.original_labels))

        with open(root.loader.count_samples_filename, "r") as fin:
            for i, n in enumerate(json.load(fin)):
                self.class_lengths[i] = n
        self.info("Class Lengths: %s", str(self.class_lengths))

        with open(root.loader.matrixes_filename, "rb") as fin:
            matrixes = pickle.load(fin)

        self.mean.mem = matrixes[0]
        self.rdisp.mem = matrixes[1].astype(
            opencl_types.dtypes[root.common.precision_type])
        if numpy.count_nonzero(numpy.isnan(self.rdisp.mem)):
            raise ValueError("rdisp matrix has NaNs")
        if numpy.count_nonzero(numpy.isinf(self.rdisp.mem)):
            raise ValueError("rdisp matrix has Infs")
        if self.mean.shape != self.rdisp.shape:
            raise ValueError("mean.shape != rdisp.shape")
        if self.mean.shape[0] != self.sy or self.mean.shape[1] != self.sx:
            raise ValueError("mean.shape != (%d, %d)" % (self.sy, self.sx))

        self.file_samples = open(root.loader.samples_filename, "rb")

    def create_minibatches(self):
        sh = [self.max_minibatch_size]
        sh.extend(self.mean.shape)
        self.minibatch_data.mem = numpy.zeros(sh, dtype=numpy.uint8)
        sh = [self.max_minibatch_size]
        self.minibatch_labels.mem = numpy.zeros(sh, dtype=numpy.int32)
        self.minibatch_indices.mem = numpy.zeros(self.max_minibatch_size,
                                                 dtype=numpy.int32)

    def fill_indices(self, start_offset, count):
        self.minibatch_indices.map_invalidate()
        idxs = self.minibatch_indices.mem
        self.shuffled_indices.map_read()
        idxs[:count] = self.shuffled_indices[start_offset:start_offset + count]

        self.minibatch_data.map_invalidate()
        self.minibatch_labels.map_invalidate()

        sample_bytes = self.mean.mem.nbytes

        for i, ii in enumerate(idxs[:count]):
            self.file_samples.seek(int(ii) * sample_bytes)
            self.file_samples.readinto(self.minibatch_data.mem[i])
            self.minibatch_labels.mem[i] = self.original_labels[int(ii)]

        if count < len(idxs):
            idxs[count:] = self.class_lengths[1]  # no data sample is there
            self.minibatch_data.mem[count:] = self.mean.mem
            self.minibatch_labels.mem[count:] = 0  # 0 is no data

        return True

    def fill_minibatch(self):
        raise error.Bug("Control should not go here")


class Workflow(StandardWorkflow):
    """Workflow.
    """
    def fix(self, unit, *attrs):
        fix = {}
        for attr in attrs:
            fix[attr] = id(getattr(unit, attr))
        self.fixed[unit] = fix

    def check_fixed(self):
        for unit, fix in self.fixed.items():
            for attr, addr in fix.items():
                if id(getattr(unit, attr)) != addr:
                    raise ValueError("Fixed attribute has changed: %s.%s" %
                                     (unit.__class__.__name__, attr))

    def init_unpickled(self):
        super(Workflow, self).init_unpickled()
        self.forward_map = {
            "conv": conv.Conv,
            "stochastic_abs_pooling": pooling.StochasticAbsPooling,
            "max_abs_pooling": pooling.MaxAbsPooling,
            "stochastic_pooling": pooling.StochasticPooling,
            "max_pooling": pooling.MaxPooling,
            "activation_mul": activation.ForwardMul,
            "all2all": all2all.All2All,
            "all2all_tanh": all2all.All2AllTanh,
            "activation_tanhlog": activation.ForwardTanhLog,
            "softmax": all2all.All2AllSoftmax}
        self.de_map = {
            conv.Conv: deconv.Deconv,
            pooling.StochasticAbsPooling: depooling.Depooling,
            pooling.MaxAbsPooling: depooling.Depooling,
            pooling.StochasticPooling: depooling.Depooling,
            pooling.MaxPooling: depooling.Depooling}
        self.gd_map = {
            deconv.Deconv: gd_deconv.GDDeconv,
            all2all.All2All: gd.GradientDescent,
            all2all.All2AllTanh: gd.GDTanh,
            all2all.All2AllSoftmax: gd.GDSM,
            activation.ForwardTanhLog: activation.BackwardTanhLog,
            activation.ForwardMul: activation.BackwardMul,
            pooling.StochasticAbsPooling: gd_pooling.GDMaxAbsPooling,
            pooling.StochasticPooling: gd_pooling.GDMaxPooling,
            pooling.MaxAbsPooling: gd_pooling.GDMaxAbsPooling,
            pooling.MaxPooling: gd_pooling.GDMaxPooling,
            conv.Conv: gd_conv.GradientDescentConv}
        self.fixed = {}

    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, minibatch_size=root.loader.minibatch_size)
        self.loader.link_from(self.repeater)
        self.fix(self.loader, "minibatch_data", "mean", "rdisp",
                 "class_lengths")

        self.meandispnorm = MeanDispNormalizer(self)
        self.meandispnorm.link_attrs(self.loader,
                                     ("input", "minibatch_data"),
                                     "mean", "rdisp")
        self.meandispnorm.link_from(self.loader)
        self.fix(self.meandispnorm, "input", "output", "mean", "rdisp")
        prev = self.meandispnorm

        ae = []
        ae_layers = []
        last_conv = None
        self.n_ae = 0
        in_ae = False
        for layer in layers:
            if layer["type"] == "ae_begin":
                self.info("Autoencoder block begin")
                in_ae = True
                continue
            if layer["type"] == "ae_end":
                self.info("Autoencoder block end")
                self.info("One AE at a time, so skipping other layers")
                self.n_ae += 1
                break
            if layer["type"][:4] == "conv":
                if in_ae:
                    layer["include_bias"] = False
                layer["padding"] = deconv.Deconv.compute_padding(
                    self.loader.sx, self.loader.sy,
                    layer["kx"], layer["ky"], layer["sliding"])
            Forward = self.forward_map[layer["type"]]
            unit = Forward(self, **layer)
            unit.layer = dict(layer)
            if in_ae:
                ae.append(unit)
                ae_layers.append(layer)
            self.fwds.append(unit)
            unit.link_from(prev)
            unit.link_attrs(prev, ("input", "output"))
            if isinstance(unit, activation.ForwardMul):
                unit.link_attrs(prev, "output")
            self.fix(unit, "input", "output", "weights")
            prev = unit
            if layer["type"][:4] == "conv" and in_ae:
                last_conv = prev
        else:
            raise error.BadFormatError("No autoencoder layers found")

        if last_conv is None:
            raise error.BadFormatError("No convolutional layer found")

        de = []
        for i in range(len(ae) - 1, -1, -1):
            De = self.de_map[ae[i].__class__]
            unit = De(self, **ae_layers[i])
            de.append(unit)
            self.fwds.append(unit)
            unit.link_from(prev)
            for dst_src in (("weights", "weights"),
                            ("get_output_shape_from", "input"),
                            ("output_offset", "input_offset")):
                if hasattr(unit, dst_src[0]):
                    unit.link_attrs(ae[i], dst_src)
            unit.link_attrs(prev, ("input", "output"))
            self.fix(unit, "input", "weights", "output",
                     "get_output_shape_from")
            prev = unit

        assert len(ae) == len(de)

        # Add evaluator unit
        unit = evaluator.EvaluatorMSE(self)
        self.evaluator = unit
        unit.link_from(self.fwds[-1])
        unit.link_attrs(self.fwds[-1], "output")
        unit.link_attrs(self.loader, ("batch_size", "minibatch_size"))
        unit.link_attrs(self.meandispnorm, ("target", "output"))
        self.fix(self.evaluator, "output", "target", "err_output", "metrics")

        # Add decision unit
        unit = decision.DecisionMSE(
            self, fail_iterations=root.decision.fail_iterations,
            max_epochs=root.decision.max_epochs)
        self.decision = unit
        unit.link_from(self.evaluator)
        unit.link_attrs(self.loader, "minibatch_class",
                        "minibatch_size", "last_minibatch",
                        "class_lengths", "epoch_ended",
                        "epoch_number")
        unit.link_attrs(self.evaluator, ("minibatch_metrics", "metrics"))
        self.fix(self.decision, "minibatch_metrics", "class_lengths")

        unit = NNSnapshotter(self, prefix=root.snapshotter.prefix,
                             directory=root.common.snapshot_dir,
                             compress="", time_interval=0)
        self.snapshotter = unit
        unit.link_from(self.decision)
        unit.link_attrs(self.decision, ("suffix", "snapshot_suffix"))
        unit.gate_skip = ~self.loader.epoch_ended | ~self.decision.improved

        unit = NNRollback(self)
        self.rollback = unit
        unit.link_from(self.snapshotter)
        unit.improved = self.decision.train_improved
        unit.gate_block = self.decision.complete
        unit.gate_skip = ~self.loader.epoch_ended

        self.end_point.link_from(self.snapshotter)
        self.end_point.gate_block = ~self.decision.complete

        # Add gradient descent unit
        GD = self.gd_map[self.fwds[-1].__class__]
        unit = GD(self, **ae_layers[0])
        self.gds.append(unit)
        unit.link_attrs(self.evaluator, "err_output")
        unit.link_attrs(self.fwds[-1], "weights", "input")
        unit.gate_skip = self.decision.gd_skip
        self.fix(unit, "err_output", "weights", "input", "err_input")
        self.rollback.add_gd(unit)

        assert len(self.gds) == 1

        self.gds[0].need_err_input = False
        self.repeater.link_from(self.gds[0])

        self.loader.gate_block = self.decision.complete

        prev = self.add_plotters(self.rollback, last_conv, ae[-1])

        self.gds[-1].link_from(prev)
        self.gds[-1].gate_block = self.decision.complete

    def del_plotters(self):
        if hasattr(self, "plt"):
            for p in self.plt:
                p.unlink_all()
                self.del_ref(p)
            del self.plt
        if hasattr(self, "plt_mx"):
            self.plt_mx.unlink_all()
            self.del_ref(self.plt_mx)
            del self.plt_mx
        if hasattr(self, "plt_inp"):
            self.plt_inp.unlink_all()
            self.del_ref(self.plt_inp)
            del self.plt_inp
        if hasattr(self, "plt_out"):
            self.plt_out.unlink_all()
            self.del_ref(self.plt_out)
            del self.plt_out
        if hasattr(self, "plt_deconv"):
            self.plt_deconv.unlink_all()
            self.del_ref(self.plt_deconv)
            del self.plt_deconv

    def add_plotters(self, prev, last_conv, last_ae):
        self.del_plotters()

        # MSE plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(prev)
            self.plt[-1].gate_skip = ~self.decision.epoch_ended
            prev = self.plt[-1]
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True

        # Weights plotter
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="Conv Weights", limit=96)
        self.plt_mx.link_attrs(last_conv, ("input", "weights"))
        self.plt_mx.get_shape_from = [last_conv.kx, last_conv.ky,
                                      last_conv.input]
        self.plt_mx.link_from(prev)
        self.plt_mx.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_mx

        # Input plotter
        self.plt_inp = nn_plotting_units.Weights2D(
            self, name="Conv Input", limit=20)
        self.plt_inp.link_attrs(last_conv, "input")
        self.plt_inp.link_from(prev)
        self.plt_inp.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_inp

        # Output plotter
        self.plt_out = nn_plotting_units.Weights2D(
            self, name="Output", limit=96)
        self.plt_out.link_attrs(last_ae, ("input", "output"))
        self.plt_out.link_from(prev)
        self.plt_out.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_out

        # Deconv result plotter
        self.plt_deconv = nn_plotting_units.Weights2D(
            self, name="Deconv result", limit=20)
        self.plt_deconv.link_attrs(self.fwds[-1], ("input", "output"))
        self.plt_deconv.link_from(prev)
        self.plt_deconv.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_deconv

        return prev

    def initialize(self, device, **kwargs):
        if (self.fwds[0].weights.mem is not None and
                root.imagenet.from_snapshot_add_layer):
            self.info("Restoring from snapshot detected, "
                      "will adjust the workflow")
            self.adjust_workflow()
            self.info("Workflow adjusted, will initialize now")
        super(Workflow, self).initialize(device, **kwargs)
        self.check_fixed()
        self.dump_attributes()

    def dump_attributes(self):
        print("Dumping the workflow unit's attributes:")
        table = prettytable.PrettyTable("#", "unit", "attr", "value")
        table.align["#"] = "r"
        table.align["unit"] = "l"
        table.align["attr"] = "l"
        table.align["value"] = "l"
        table.max_width["value"] = 100
        for i, u in enumerate(self.start_point.dependent_list()):
            for k, v in sorted(u.__dict__.items()):
                table.add_row(i, u.__class__.__name__, k, repr(v))
        print(str(table))

    def switch_to_fine_tuning(self):
        if len(self.gds) == len(self.fwds):
            self.info("Already at fine-tune stage, will exit with code 1 now")
            os._exit(1)
        # Add gradient descent units for the remaining forward units
        self.gds[0].unlink_after()
        self.gds[0].need_err_input = True
        prev = self.gds[0]

        for i in range(len(self.fwds) - len(self.gds) - 1, -1, -1):
            GD = self.gd_map[self.fwds[i].__class__]
            kwargs = dict(self.fwds[i].layer)
            for attr in ("n_kernels", "kx", "ky", "sliding", "padding",
                         "factor", "include_bias"):
                vle = getattr(self.fwds[i], attr, None)
                if vle is not None:
                    kwargs[attr] = vle
            kwargs["learning_rate"] = kwargs["learning_rate_ft"]
            if "learning_rate_ft_bias" in kwargs:
                kwargs["learning_rate_bias"] = kwargs["learning_rate_ft_bias"]
            unit = GD(self, **kwargs)
            self.gds.insert(0, unit)
            unit.link_from(prev)
            unit.link_attrs(prev, ("err_output", "err_input"))
            if hasattr(self.fwds[i], "input_offset"):
                unit.link_attrs(self.fwds[i], "weights", "input", "output",
                                "input_offset")
            else:
                unit.link_attrs(self.fwds[i], "weights", "input", "output")
            if self.fwds[i].bias is None:
                unit.link_attrs(self.fwds[i], "bias")
            unit.gate_skip = self.decision.gd_skip
            prev = unit
            self.fix(unit, "weights", "input", "output",
                     "err_input", "err_output")

        self.gds[0].need_err_input = False
        self.repeater.link_from(self.gds[0])

        self.rollback.reset()
        noise = float(root.imagenet.fine_tuning_noise)
        for unit in self.gds:
            self.rollback.add_gd(unit)
            if not noise:
                continue
            if unit.weights:
                a = unit.weights.plain
                a += prng.get().normal(0, noise, unit.weights.size)
            if unit.bias:
                a = unit.bias.plain
                a += prng.get().normal(0, noise, unit.bias.size)

        # Reset last best error, `cause we have modified the workflow
        self.decision.min_validation_n_err = 1.0e30
        self.decision.min_train_validation_n_err = 1.0e30
        self.decision.min_train_n_err = 1.0e30

    def adjust_workflow(self):
        self.info("Will extend %d autoencoder layers", self.n_ae)

        layers = root.imagenet.layers
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
                if n_ae >= self.n_ae:
                    break
                continue
            i_fwd += 1
        else:
            self.warning("Will switch to the fine-tuning task")
            return self.switch_to_fine_tuning()

        i_fwd = i_fwd_last
        for i in range(i_fwd, len(self.fwds)):
            self.fwds[i].unlink_all()
            self.del_ref(self.fwds[i])
        del self.fwds[i_fwd:]
        last_fwd = self.fwds[-1]
        prev = last_fwd

        ae = []
        ae_layers = []
        last_conv = None
        in_ae = False
        for layer in layers[i_layer:]:
            if layer["type"] == "ae_begin":
                self.info("Autoencoder block begin")
                in_ae = True
                continue
            if layer["type"] == "ae_end":
                self.info("Autoencoder block end")
                self.info("One AE at a time, so skipping other layers")
                self.n_ae += 1
                break
            if layer["type"][:4] == "conv":
                if in_ae:
                    layer["include_bias"] = False
                layer["padding"] = deconv.Deconv.compute_padding(
                    last_fwd.output.shape[2], last_fwd.output.shape[1],
                    layer["kx"], layer["ky"], layer["sliding"])
                self.info("Computed padding for (kx, ky)=(%d, %d) "
                          "sliding=%s is %s", layer["kx"], layer["ky"],
                          str(layer["sliding"]), str(layer["padding"]))
            Forward = self.forward_map[layer["type"]]
            unit = Forward(self, **layer)
            unit.layer = dict(layer)
            if in_ae:
                ae.append(unit)
                ae_layers.append(layer)
            self.fwds.append(unit)
            unit.link_from(prev)
            unit.link_attrs(prev, ("input", "output"))
            if isinstance(unit, activation.ForwardMul):
                unit.link_attrs(prev, "output")
            self.fix(unit, "input", "output", "weights")
            prev = unit
            if layer["type"][:4] == "conv" and in_ae:
                last_conv = prev

        if last_conv is None and in_ae:
            raise error.BadFormatError("No convolutional layer found")

        if in_ae:
            de = []
            for i in range(len(ae) - 1, -1, -1):
                De = self.de_map[ae[i].__class__]
                unit = De(self, **ae_layers[i])
                de.append(unit)
                self.fwds.append(unit)
                unit.link_from(prev)
                for dst_src in (("weights", "weights"),
                                ("get_output_shape_from", "input"),
                                ("output_offset", "input_offset")):
                    if hasattr(unit, dst_src[0]):
                        unit.link_attrs(ae[i], dst_src)
                unit.link_attrs(prev, ("input", "output"))
                self.fix(unit, "input", "weights", "output",
                         "get_output_shape_from")
                prev = unit

            unit = self.evaluator
            unit.link_from(self.fwds[-1])
            unit.link_attrs(self.fwds[-1], "output")
            unit.link_attrs(last_conv, ("target", "input"))
            self.fix(self.evaluator, "output", "target", "err_output",
                     "metrics")

            assert len(self.gds) == 1

            self.gds[0].unlink_all()
            self.del_ref(self.gds[0])
            del self.gds[:]

            # Add gradient descent unit
            GD = self.gd_map[self.fwds[-1].__class__]
            unit = GD(self, **ae_layers[0])
            self.gds.append(unit)
            unit.link_attrs(self.evaluator, "err_output")
            unit.link_attrs(self.fwds[-1], "weights", "input")
            unit.gate_skip = self.decision.gd_skip
            self.fix(unit, "err_output", "weights", "input", "err_input")
            self.rollback.reset()
            self.rollback.add_gd(unit)

            assert len(self.gds) == 1

            self.gds[0].need_err_input = False
            self.repeater.link_from(self.gds[0])

            prev = self.add_plotters(self.rollback, last_conv, ae[-1])

            self.gds[-1].link_from(prev)
            self.gds[-1].gate_block = self.decision.complete

            # Reset last best error, `cause we have extended the workflow
            self.decision.min_validation_mse = 1.0e30
            self.decision.min_train_validation_mse = 1.0e30
            self.decision.min_train_mse = 1.0e30
            self.decision.min_validation_n_err = 1.0e30
            self.decision.min_train_validation_n_err = 1.0e30
            self.decision.min_train_n_err = 1.0e30
        else:
            self.info("No more autoencoder levels, "
                      "will switch to the classification task")
            self.n_ae += 1

            # Add evaluator unit
            self.evaluator.unlink_all()
            self.del_ref(self.evaluator)
            self.evaluator = None
            unit = evaluator.EvaluatorSoftmax(self)
            self.evaluator = unit
            unit.link_from(self.fwds[-1])
            unit.link_attrs(self.fwds[-1], "output", "max_idx")
            unit.link_attrs(self.loader, ("labels", "minibatch_labels"),
                            ("batch_size", "minibatch_size"))
            self.fix(self.evaluator, "output", "max_idx",
                     "labels", "err_output")

            # Add decision unit
            self.decision.unlink_all()
            self.del_ref(self.decision)
            self.decision = None
            unit = decision.DecisionGD(
                self, fail_iterations=root.decision.fail_iterations)
            self.decision = unit
            unit.link_from(self.evaluator)
            unit.link_attrs(self.loader, "minibatch_class",
                            "minibatch_size", "last_minibatch",
                            "class_lengths", "epoch_ended",
                            "epoch_number")
            unit.link_attrs(self.evaluator, ("minibatch_n_err", "n_err"),
                            ("minibatch_confusion_matrix", "confusion_matrix"),
                            ("minibatch_max_err_y_sum", "max_err_output_sum"))
            self.fix(self.decision, "minibatch_n_err", "class_lengths")

            unit = self.snapshotter
            unit.link_from(self.decision)
            unit.link_attrs(self.decision, ("suffix", "snapshot_suffix"))
            unit.gate_skip = ~self.loader.epoch_ended | ~self.decision.improved

            self.rollback.gate_block = self.decision.complete
            self.rollback.improved = self.decision.train_improved

            assert len(self.gds) == 1

            self.gds[0].unlink_all()
            self.del_ref(self.gds[0])
            del self.gds[:]

            # Add gradient descent units
            self.rollback.reset()
            prev_gd = self.evaluator
            prev = None
            gds = []
            for i in range(len(self.fwds) - 1, i_fwd - 1, -1):
                GD = self.gd_map[self.fwds[i].__class__]
                kwargs = {}
                for attr in ("n_kernels", "kx", "ky", "sliding", "padding",
                             "factor", "include_bias"):
                    vle = getattr(self.fwds[i], attr, None)
                    if vle is not None:
                        kwargs[attr] = vle
                unit = GD(self, **kwargs)
                gds.append(unit)
                if prev is not None:
                    unit.link_from(prev)
                if isinstance(prev_gd, evaluator.EvaluatorBase):
                    unit.link_attrs(prev_gd, "err_output")
                else:
                    unit.link_attrs(prev_gd, ("err_output", "err_input"))
                if hasattr(self.fwds[i], "input_offset"):
                    unit.link_attrs(self.fwds[i], "weights", "bias",
                                    "input", "output", "input_offset")
                else:
                    unit.link_attrs(self.fwds[i], "weights", "bias",
                                    "input", "output")
                unit.gate_skip = self.decision.gd_skip
                prev_gd = unit
                prev = unit
                self.fix(unit, "weights", "bias", "input", "output",
                         "err_input", "err_output")
                self.rollback.add_gd(unit)
            # Strip gd's without weights
            for i in range(len(gds) - 1, -1, -1):
                if (isinstance(gds[i], gd.GradientDescent) or
                        isinstance(gds[i], gd_conv.GradientDescentConv)):
                    break
                unit = gds.pop(-1)
                unit.unlink_all()
                self.del_ref(unit)
                del self.fixed[unit]
            self.gds = list(None for _ in gds)
            for i, _gd in enumerate(gds):
                self.gds[-(i + 1)] = _gd
            del gds

            self.gds[0].need_err_input = False
            self.repeater.link_from(self.gds[0])

            self.del_plotters()

            # Error plotter
            prev = self.rollback
            self.plt = []
            styles = ["r-", "b-", "k-"]
            for i in range(1, 3):
                self.plt.append(plotting_units.AccumulatingPlotter(
                    self, name="Errors", plot_style=styles[i]))
                self.plt[-1].input = self.decision.epoch_n_err_pt
                self.plt[-1].input_field = i
                self.plt[-1].link_from(prev)
                self.plt[-1].gate_skip = ~self.decision.epoch_ended
                prev = self.plt[-1]
            self.plt[0].clear_plot = True
            self.plt[-1].redraw_plot = True

            self.gds[-1].link_from(prev)
            self.gds[-1].gate_block = self.decision.complete


def run(load, main):
    root.snapshotter.prefix = "imagenet_ae_%s" % root.loader.year
    CACHED_DATA_FNME = os.path.join(IMAGENET_BASE_PATH, str(root.loader.year))
    root.loader.names_labels_filename = os.path.join(
        CACHED_DATA_FNME, "original_labels_%s_%s_0.pickle" %
        (root.loader.year, root.loader.series))
    root.loader.count_samples_filename = os.path.join(
        CACHED_DATA_FNME, "count_samples_%s_%s_0.json" %
        (root.loader.year, root.loader.series))
    root.loader.samples_filename = os.path.join(
        CACHED_DATA_FNME, "original_data_%s_%s_0.dat" %
        (root.loader.year, root.loader.series))
    root.loader.matrixes_filename = os.path.join(
        CACHED_DATA_FNME, "matrixes_%s_%s_0.pickle" %
        (root.loader.year, root.loader.series))
    load(Workflow, layers=root.imagenet.layers)
    main()
