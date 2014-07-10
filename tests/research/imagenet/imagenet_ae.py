#!/usr/bin/python3.3 -O

"""
Created on July 4, 2014

Imagenet recognition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


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
import veles.znicz.pooling as pooling
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.nn_units import NNSnapshotter
from veles.znicz.standard_workflow import StandardWorkflow
from veles.mean_disp_normalizer import MeanDispNormalizer

IMAGENET_BASE_PATH = os.path.join(root.common.test_dataset_root,
                                  "imagenet")
root.model = "imagenet"

LR = 0.0000001
LRB = LR
WD = 0.004
WDB = WD
GM = 0.9
GMB = GM

root.defaults = {
    "decision": {"fail_iterations": 100000,
                 "use_dynamic_alpha": False,
                 "do_export_weights": True},
    "snapshotter": {"prefix": "imagenet"},
    "loader": {"year": "temp",
               "series": "img",
               "minibatch_size": 20},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir,
                                  "tmp %s/test" % root.model),
                     os.path.join(root.common.cache_dir,
                                  "tmp %s/validation" % root.model),
                     os.path.join(root.common.cache_dir,
                                  "tmp %s/train" % root.model)]},
    "weights_plotter": {"limit": 64},
    "imagenet": {"layers":
                 [{"type": "conv", "n_kernels": 96,
                   "kx": 11, "ky": 11, "padding": (5, 5, 5, 5),
                   "sliding": (1, 1),
                   "learning_rate": LR,
                   "weights_decay": WD,
                   "gradient_moment": GM},
                  {"type": "stochastic_abs_pooling",
                   "kx": 3, "ky": 3, "sliding": (2, 2)},

                  {"type": "softmax", "output_shape": 5,
                   "learning_rate": LR, "learning_rate_bias": LRB,
                   "weights_decay": WD, "weights_decay_bias": WDB,
                   "gradient_moment": GM, "gradient_moment_bias": GMB}]}}

CACHED_DATA_FNME = os.path.join(IMAGENET_BASE_PATH, root.loader.year)
root.loader.names_labels_filename = os.path.join(
    CACHED_DATA_FNME, "4/names_labels_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))
root.loader.count_samples_filename = os.path.join(
    CACHED_DATA_FNME, "4/count_samples_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))
root.loader.samples_filename = os.path.join(
    CACHED_DATA_FNME, "4/original_data_%s_%s_0.dat" %
    (root.loader.year, root.loader.series))
root.loader.matrixes_filename = os.path.join(
    CACHED_DATA_FNME, "4/matrixes_%s_%s_0.pickle" %
    (root.loader.year, root.loader.series))


@implementer(loader.ILoader)
class Loader(loader.Loader):
    """loads imagenet from samples.dat, labels.pickle"""
    def __init__(self, workflow, **kwargs):
        super(Loader, self).__init__(workflow, **kwargs)
        self.mean = Vector()
        self.rdisp = Vector()
        self.file_samples = ""

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

        with open(root.loader.count_samples_filename, "rb") as fin:
            for i, n in enumerate(pickle.load(fin)):
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

    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.saver = None
        self.fixed = {}

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

        layer_conv = layers[0]
        layer_conv["include_bias"] = False
        unit = conv.Conv(self, **layer_conv)
        unit.link_from(self.meandispnorm)
        unit.link_attrs(self.meandispnorm, ("input", "output"))
        self.conv = unit
        self.fix(self.conv, "input", "output", "weights")

        layer_pool = layers[1]
        unit = pooling.StochasticAbsPooling(self, **layer_pool)
        unit.link_from(self.conv)
        unit.link_attrs(self.conv, ("input", "output"))
        self.pool = unit
        self.fix(self.pool, "input", "output")

        unit = gd_pooling.GDMaxAbsPooling(self, **layer_pool)
        unit.link_from(self.pool)
        unit.link_attrs(self.pool, "input", "input_offset",
                        ("err_output", "output"))
        self.depool = unit
        self.fix(self.depool, "input", "err_output", "err_input")

        unit = deconv.Deconv(self, **layer_conv)
        self.deconv = unit
        unit.link_from(self.depool)
        unit.link_attrs(self.conv, "weights")
        unit.link_attrs(self.depool, ("input", "err_input"))
        self.fix(self.deconv, "input", "weights", "output")

        # Add evaluator for single minibatch
        unit = evaluator.EvaluatorMSE(self)
        self.evaluator = unit
        unit.link_from(self.deconv)
        unit.link_attrs(self.deconv, "output")
        unit.link_attrs(self.loader, ("batch_size", "minibatch_size"))
        unit.link_attrs(self.meandispnorm, ("target", "output"))
        self.fix(self.evaluator, "output", "target", "err_output", "metrics")

        # Add decision unit
        unit = decision.DecisionMSE(
            self, fail_iterations=root.decision.fail_iterations)
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

        self.end_point.link_from(self.snapshotter)
        self.end_point.gate_block = ~self.decision.complete

        # Add gradient descent units
        unit = gd_deconv.GDDeconv(self, **layer_conv)
        self.gd_deconv = unit
        unit.link_attrs(self.evaluator, "err_output")
        unit.link_attrs(self.deconv, "weights", "input")
        unit.gate_skip = self.decision.gd_skip
        self.fix(self.gd_deconv, "err_output", "weights", "input", "err_input")

        self.gd_deconv.need_err_input = False
        self.repeater.link_from(self.gd_deconv)

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        prev = self.snapshotter
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
            self, name="Weights", limit=96)
        self.plt_mx.link_attrs(self.conv, ("input", "weights"))
        self.plt_mx.get_shape_from = [layer_conv["kx"], layer_conv["ky"], 4]
        self.plt_mx.link_from(prev)
        self.plt_mx.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_mx

        # Input plotter
        self.plt_inp = nn_plotting_units.Weights2D(
            self, name="Conv Input", limit=20)
        self.plt_inp.link_attrs(self.conv, "input")
        self.plt_inp.get_shape_from = self.conv.input
        self.plt_inp.link_from(prev)
        self.plt_inp.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_inp

        # Output plotter
        self.plt_out = nn_plotting_units.Weights2D(
            self, name="Pooling Output", limit=96)
        self.plt_out.link_attrs(self.pool, ("input", "output"))
        self.plt_out.get_shape_from = self.pool.output
        self.plt_out.link_from(prev)
        self.plt_out.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_out

        # Deconv result plotter
        self.plt_deconv = nn_plotting_units.Weights2D(
            self, name="Deconv result", limit=20)
        self.plt_deconv.link_attrs(self.deconv, ("input", "output"))
        self.plt_deconv.get_shape_from = self.deconv.output
        self.plt_deconv.link_from(prev)
        self.plt_deconv.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_deconv

        self.gd_deconv.link_from(prev)
        self.gd_deconv.gate_block = self.decision.complete

    def initialize(self, device, **kwargs):
        super(Workflow, self).initialize(device, **kwargs)
        self.check_fixed()


def run(load, main):
    load(Workflow, layers=root.imagenet.layers)
    main()
