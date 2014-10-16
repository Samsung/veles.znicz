"""
Created on Jan 28, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import gc
import numpy
import os
import six
import sys
import tarfile
from zope.interface import implementer

from veles.distributable import IDistributable
import veles.formats as formats
from veles.mutable import Bool
from veles.opencl_units import OpenCLUnit, OpenCLWorkflow
import veles.prng as prng
from veles.workflow import Repeater
from veles.snapshotter import SnapshotterBase, Snapshotter
from veles.error import MasterSlaveCommunicationError
from veles.timeit import timeit


@implementer(IDistributable)
class Forward(OpenCLUnit):
    """Base class for forward propagation units.

    Attributes:
        input: input layer values.
        output: output layer values.
        weights: weights.
        bias: bias.
        weights_stddev: magnitude of the random distribution for weights.
        bias_stddev: magnitude of the random distribution for bias.
        rand: prng.Rand() object for initial weights generation.
    """
    hide = True

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "WORKER")
        super(Forward, self).__init__(workflow, **kwargs)
        self.weights_stddev = kwargs.get("weights_stddev")
        self.bias_stddev = kwargs.get("bias_stddev", self.weights_stddev)
        self.weights_filling = kwargs.get("weights_filling", "uniform")
        self.bias_filling = kwargs.get("bias_filling", "uniform")
        self.rand = kwargs.get("rand", prng.get())
        self.weights_transposed = kwargs.get("weights_transposed", False)
        self.include_bias = kwargs.get("include_bias", True)
        self.demand("input")
        self.output = formats.Vector()
        self.weights = formats.Vector()
        self.bias = formats.Vector()
        self.exports = ["weights", "bias",
                        "include_bias", "weights_transposed"]

    def initialize(self, device, **kwargs):
        super(Forward, self).initialize(device=device, **kwargs)
        self.forward_mode = kwargs.get("forward_mode", False)

    def generate_data_for_slave(self, slave):
        if self.forward_mode:
            return None
        data = [None, None]
        if self.weights:
            self.weights.map_read()
            data[0] = self.weights.mem
        if self.bias:
            self.bias.map_read()
            data[1] = self.bias.mem
        return data

    def generate_data_for_master(self):
        return None

    def apply_data_from_master(self, data):
        if self.forward_mode:
            return
        if self.weights:
            self.weights.map_invalidate()
            numpy.copyto(self.weights.mem, data[0])
        if self.bias:
            self.bias.map_invalidate()
            numpy.copyto(self.bias.mem, data[1])

    def apply_data_from_slave(self, data, slave):
        pass

    def drop_slave(self, slave):
        pass


@implementer(IDistributable)
class GradientDescentBase(OpenCLUnit):
    """Base class for gradient descent units.

    Attributes:
        input: input layer values.
        output: output layer values.
        err_output: error to backpropagate.
        err_input: backpropagated error.
        weights: weights.
        bias: bias.
        batch_size: current minibatch size.
        learning_rate: gradient descent speed (positive).
        learning_rate_bias: gradient descent speed for bias
        weights_decay: coefficient (positive or zero) for weights
                       regularization term (lambda/2 * sum(weights^2)).
        weights_decay_bias
        gradient_moment
        gradient_moment_bias
        batch_size: effective batch size (if None, get it from y).
        weights_transposed: assume weights matrix as a transposed one.
        store_gradient: will save gradient as separate Vector().
        apply_gradient: will apply gradient.
        gradients_changed: when True, slave will send gradients to master
            (assigned to True just before the run call, so it can be set to
            False inside ocl_run, cpu_run if necessary)
    """
    hide = True

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "TRAINER")
        super(GradientDescentBase, self).__init__(workflow, **kwargs)
        self.input = None
        self.output = None
        self.err_output = None  # formats.Vector()
        self.err_input = formats.Vector()
        self.weights = None
        self.bias = None
        self.batch_size = None
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.learning_rate_bias = kwargs.get("learning_rate_bias",
                                             self.learning_rate)
        self.weights_decay = kwargs.get("weights_decay", 0.00005)
        self.weights_decay_bias = kwargs.get("weights_decay_bias", 0.0)
        self.l1_vs_l2 = kwargs.get("l1_vs_l2", 0)
        self.l1_vs_l2_bias = kwargs.get("l1_vs_l2_bias", self.l1_vs_l2)
        self.gradient_moment = kwargs.get("gradient_moment", 0)
        self.gradient_moment_bias = kwargs.get("gradient_moment_bias",
                                               self.gradient_moment)
        self.store_gradient = kwargs.get("store_gradient", True)
        self.apply_gradient = kwargs.get("apply_gradient",
                                         not workflow.is_slave)
        self.weights_transposed = kwargs.get("weights_transposed", False)
        self.need_err_input = kwargs.get("need_err_input", True)
        self.include_bias = kwargs.get("include_bias", True)
        self.factor_ortho = kwargs.get("factor_ortho", 0)
        self.col_sums = formats.Vector()  # for orthogonalization
        self.store_gradient = bool(
            (not workflow.is_slave and
             (self.gradient_moment or self.gradient_moment_bias)) or
            self.store_gradient)
        self.gradient_weights = formats.Vector()
        self.gradient_bias = formats.Vector()
        self.gradients_changed = False

    @property
    def current_batch_size(self):
        if self.batch_size is None:
            return self.err_output.mem.shape[0]
        return int(self.batch_size)

    def initialize(self, device, **kwargs):
        super(GradientDescentBase, self).initialize(device, **kwargs)
        self.learning_rate = kwargs.get("learning_rate", self.learning_rate)
        self.weights_decay = kwargs.get("weights_decay", self.weights_decay)
        self.gradient_moment = kwargs.get("gradient_moment",
                                          self.gradient_moment)
        self.learning_rate_bias = kwargs.get("learning_rate_bias",
                                             self.learning_rate_bias)
        self.weights_decay_bias = kwargs.get("weights_decay_bias",
                                             self.weights_decay_bias)
        self.gradient_moment = kwargs.get("gradient_moment_bias",
                                          self.gradient_moment_bias)

    def generate_data_for_slave(self, slave):
        return (self.learning_rate, self.weights_decay, self.gradient_moment,
                self.learning_rate_bias, self.weights_decay_bias,
                self.gradient_moment_bias)

    def apply_data_from_master(self, data):
        self.learning_rate = data[0]
        self.weights_decay = data[1]
        self.gradient_moment = data[2]
        self.learning_rate_bias = data[3]
        self.weights_decay_bias = data[4]
        self.gradient_moment_bias = data[5]
        if self.gradient_weights:
            self.gradient_weights.map_invalidate()
            self.gradient_weights.mem[:] = 0
        if self.gradient_bias:
            self.gradient_bias.map_invalidate()
            self.gradient_bias.mem[:] = 0

    def generate_data_for_master(self):
        if not self.gradients_changed:
            return None
        self.gradients_changed = False
        self.gradient_weights.map_read()
        self.gradient_bias.map_read()
        return (self.gradient_weights.mem, self.gradient_bias.mem)

    def apply_data_from_slave(self, data, slave):
        if self.weights:
            self.weights.map_write()
            if self.store_gradient:
                self.gradient_weights.map_write()
                self.gradient_weights.mem *= self.gradient_moment
                self.gradient_weights.mem += data[0]
                self.weights.mem += self.gradient_weights.mem
            else:
                self.weights.mem += data[0]
        if self.bias:
            self.bias.map_write()
            if self.store_gradient:
                self.gradient_bias.map_write()
                self.gradient_bias.mem *= self.gradient_moment_bias
                self.gradient_bias.mem += data[1]
                self.bias.mem += self.gradient_bias.mem
            else:
                self.bias.mem += data[1]

    def drop_slave(self, slave):
        pass

    @staticmethod
    def cpu_gradient_step(weight, gradient, lr, factor_l12, l1_vs_l2,
                          factor_ortho=0):
        g = gradient.copy()
        g += factor_l12 * ((1.0 - l1_vs_l2) * weight +
                           0.5 * l1_vs_l2 * numpy.sign(weight))
        if factor_ortho:
            col_sums = weight.sum(axis=0)
            for i, row in enumerate(g):
                row += (col_sums - weight[i]) * factor_ortho / weight.shape[0]
        g *= lr
        return g

    def run(self):
        if self.store_gradient:
            self.gradients_changed = True
        super(GradientDescentBase, self).run()


class NNWorkflow(OpenCLWorkflow):
    """Base class for neural network workflow.

    Attributes:
        repeater: Repeater unit.
        loader: loader.Loader unit.
        fwds: list of the forward propagation (Forward) units.
        evaluator: evaluator.* unit.
        decision: decision.Decision unit.
        gds: list of the gradient descent units.
    """
    def __init__(self, workflow, **kwargs):
        super(NNWorkflow, self).__init__(workflow, **kwargs)
        self.repeater = Repeater(self)
        self.loader = None
        self.fwds = []
        self.evaluator = None
        self.decision = None
        self.gds = []

    def validate_history(self):
        """Raises error.MasterSlaveCommunicationError if apply-generate
        history is invalid.
        """
        if not self.is_master:
            return

        from collections import defaultdict

        self.debug("Checking the history...")
        async = self.workflow.args.async
        job_stack = defaultdict(int)
        for index, record in enumerate(self.history):
            sid = record[2]
            if record[0] == "generate":
                job_stack[sid] += 1
            elif record[0] == "apply":
                job_stack[sid] -= 1
            if async and job_stack[sid] == 0:
                raise MasterSlaveCommunicationError(
                    "Apply-generate balance becomes zero at index %d" % index)
            if not async and job_stack[sid] == 2:
                raise MasterSlaveCommunicationError(
                    "Apply-generate balance becomes 2 at index %d" % index)
        self.info("History validation's been completed. Everything is OK.")

    def export(self, file_name):
        """Exports workflow for use on DTV.
        """
        exported = [u for u in self if hasattr(u, "export")]
        if len(exported) == 0:
            raise ValueError("No units support export. Implement export() "
                             "method in at least one.")
        obj = {"workflow": self.name,
               "checksum": self.checksum(),
               "units": [{"class": {"name": unit.__class__.__name__,
                                    "uuid": unit.__class__.__id__},
                          "data": unit.export()}
                         for unit in exported]}
        for index, unit in enumerate(exported):
            obj["units"][index]["links"] = [
                exported.index(u) for u in sorted(unit.links_to.keys())
                if u in exported]
        # TODO(v.markovtsev): check the resulting graph's connectivity
        # TODO(v.markovtsev): check for single entry and exit points

        import json

        arrays = []

        def array_file_name(arr, index):
            return "%04d_%s" % (index, "x".join(arr.shape))

        def export_numpy_array(arr):
            if isinstance(arr, numpy.ndarray):
                arrays.append(arr)
                return array_file_name(arr, len(arrays) - 1)
            raise TypeError("Objects of class other than numpy.ndarray are "
                            "not supported")
        try:
            with tarfile.open(file_name, "w:gz") as tar:
                io = six.BytesIO()
                json.dump(obj, io, indent=4, sort_keys=True,
                          default=export_numpy_array)
                ti = tarfile.TarInfo("contents.json")
                ti.size = io.tell()
                ti.mode = int("666", 8)
                io.seek(0)
                tar.addfile(ti, fileobj=io)
                for index, arr in enumerate(arrays):
                    io = six.BytesIO()
                    numpy.save(io, arr)
                    ti = tarfile.TarInfo(array_file_name(arr, index) + ".npy")
                    ti.size = io.tell()
                    ti.mode = int("666", 8)
                    io.seek(0)
                    tar.addfile(ti, fileobj=io)
        except:
            self.exception("Failed to export to %s", file_name)


class ForwardExporter(SnapshotterBase):
    """Saves weights and biases from Forward units.

    Defines:
        file_name - the file name of the last export.
        time - the time of the last export

    Must be defined before initialize():
        suffix - the file name suffix where to export weights and biases
        forwards - the list of Forward units to take weights and biases from

    Attributes:
        compress - the compression applied to pickles: None or '', gz, bz2, xz
        compress_level - the compression level in [0..9]
        interval - take only one snapshot within this run() invocation number
        time_interval - take no more than one snapshot within this time window
    """

    CODECS = {
        None: lambda n, l: tarfile.TarFile.open(n, "w"),
        "": lambda n, l: tarfile.TarFile.open(n, "w"),
        "gz": lambda n, l: tarfile.TarFile.gzopen(n, "w", compresslevel=l),
        "bz2": lambda n, l: tarfile.TarFile.bz2open(n, "w", compresslevel=l),
        "xz": lambda n, l: tarfile.TarFile.xzopen(n, "w", preset=l)
    }

    def __init__(self, workflow, **kwargs):
        super(ForwardExporter, self).__init__(workflow, **kwargs)
        self.forwards = []

    def export(self):
        ext = ("." + self.compress) if self.compress else ""
        rel_file_name = "%s_%s_wb.%d.tar%s" % (
            self.prefix, self.suffix, sys.version_info[0], ext)
        self.file_name = os.path.join(self.directory, rel_file_name)
        with self._open_file() as tar:
            for index, fwd in enumerate(self.forwards):
                weights, bias = fwd.generate_data_for_slave(None)
                fileobj = six.BytesIO()
                numpy.savez(fileobj, weights=weights, bias=bias)
                ti = tarfile.TarInfo("%03d_%s.npz" % (index + 1, fwd.name))
                ti.size = fileobj.tell()
                ti.mode = int("666", 8)
                fileobj.seek(0)
                tar.addfile(ti, fileobj=fileobj)
        self.info("Wrote %s" % self.file_name)
        file_name_link = os.path.join(
            self.directory, "%s_current_wb.%d.tar%s" % (
                self.prefix, sys.version_info[0], ext))
        if os.path.exists(file_name_link):
            os.remove(file_name_link)
        os.symlink(rel_file_name, file_name_link)

    def _open_file(self):
        return ForwardExporter.CODECS[self.compress](self.file_name,
                                                     self.compress_level)


class NNSnapshotter(Snapshotter):
    def __init__(self, workflow, **kwargs):
        super(NNSnapshotter, self).__init__(workflow, **kwargs)
        self.has_invalid_values = Bool(False)

    def _log_attr(self, unit, attr, logged):
        val = getattr(unit, attr, None)
        if val is None:
            return
        mem = getattr(val, "mem", None)
        if mem is None:
            return
        if id(mem) not in logged:
            self.has_invalid_values <<= bool(
                numpy.count_nonzero(numpy.isnan(mem)) or
                numpy.count_nonzero(numpy.isinf(mem)))
            args = ("%s: %s: min max avg: %.6f %.6f %.6f%s",
                    unit.__class__.__name__, attr,
                    mem.min(), mem.max(), numpy.average(mem),
                    " has invalid values" if self.has_invalid_values else "")
            if self.has_invalid_values:
                self.error(*args)
            else:
                self.info(*args)
            logged.add(id(mem))

    def export(self):
        super(NNSnapshotter, self).export()
        logged = set()
        for u in self.workflow.start_point.dependent_list():
            for attr in ("input", "weights", "bias", "output",
                         "err_output", "err_input"):
                self._log_attr(u, attr, logged)
        del logged
        _, dt = timeit(gc.collect)
        if dt > 1.0:
            self.warning("gc.collect() took %.1f sec", dt)
