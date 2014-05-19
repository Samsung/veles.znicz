"""
Created on Jan 28, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import shutil
import six
import tarfile
import yaml

import veles.config as config
import veles.formats as formats
from veles.opencl_units import OpenCLUnit, OpenCLWorkflow
import veles.random_generator as rnd
from veles.units import Repeater
from veles.snapshotter import SnapshotterBase


class Forward(OpenCLUnit):
    """Base class for forward propagation units.

    Attributes:
        input: input layer values.
        output: output layer values.
        weights: weights.
        bias: bias.
        weights_stddev: magnitude of the random distribution for weights.
        bias_stddev: magnitude of the random distribution for bias.
        rand: rnd.Rand() object for initial weights generation.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "WORKER")
        super(Forward, self).__init__(workflow, **kwargs)
        self.weights_stddev = kwargs.get("weights_stddev")
        self.bias_stddev = kwargs.get("bias_stddev", self.weights_stddev)
        self.weights_filling = kwargs.get("weights_filling", "uniform")
        self.bias_filling = kwargs.get("bias_filling", "uniform")
        self.rand = kwargs.get("rand", rnd.get())
        self.weights_transposed = kwargs.get("weights_transposed", False)
        self.input = None
        self.output = formats.Vector()
        self.weights = formats.Vector()
        self.bias = formats.Vector()
        self.exports = ["weights", "bias", "weights_transposed"]

    def generate_data_for_slave(self, slave=None):
        self.weights.map_read()
        self.bias.map_read()
        data = (self.weights.v.copy(), self.bias.v.copy())
        return data

    def apply_data_from_master(self, data):
        self.weights.map_invalidate()
        self.bias.map_invalidate()
        numpy.copyto(self.weights.v, data[0])
        numpy.copyto(self.bias.v, data[1])


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
        weights_decay: coefficient (positive or zero) for weights
                       regularization term (lambda/2 * sum(weights^2)).
        batch_size: effective batch size (if None, get it from y).
        weights_transposed: assume weights matrix as a transposed one.
        store_gradient: will save gradient as separate Vector().
        apply_gradient: will apply gradient.
    """
    def __init__(self, workflow, **kwargs):
        learning_rate = kwargs.get("learning_rate", 0.01)
        learning_rate_bias = kwargs.get("learning_rate_bias", learning_rate)
        weights_decay = kwargs.get("weights_decay", 0.00005)
        weights_decay_bias = kwargs.get("weights_decay_bias", 0.0)
        weights_transposed = kwargs.get("weights_transposed", False)
        gradient_moment = kwargs.get("gradient_moment", 0)
        gradient_moment_bias = kwargs.get("gradient_moment_bias",
                                          gradient_moment)
        store_gradient = kwargs.get("store_gradient", workflow.is_slave)
        apply_gradient = kwargs.get("apply_gradient", not workflow.is_slave)
        need_err_input = kwargs.get("need_err_input", True)
        kwargs["learning_rate"] = learning_rate
        kwargs["learning_rate_bias"] = learning_rate_bias
        kwargs["weights_decay"] = weights_decay
        kwargs["weights_decay_bias"] = weights_decay_bias
        kwargs["weights_transposed"] = weights_transposed
        kwargs["store_gradient"] = store_gradient
        kwargs["apply_gradient"] = apply_gradient
        kwargs["need_err_input"] = need_err_input
        kwargs["gradient_moment"] = gradient_moment
        kwargs["gradient_moment_bias"] = gradient_moment_bias
        kwargs["view_group"] = kwargs.get("view_group", "TRAINER")
        super(GradientDescentBase, self).__init__(workflow, **kwargs)
        self.input = None
        self.output = None
        self.err_output = None  # formats.Vector()
        self.err_input = formats.Vector()
        self.weights = None
        self.bias = None
        self.batch_size = None
        self.learning_rate = learning_rate
        self.learning_rate_bias = learning_rate_bias
        self.weights_decay = weights_decay
        self.weights_decay_bias = weights_decay_bias
        self.weights_transposed = weights_transposed
        self.gradient_moment = gradient_moment
        self.gradient_moment_bias = gradient_moment_bias
        self.store_gradient = ((not workflow.is_slave and
                                (gradient_moment or gradient_moment_bias)) or
                               store_gradient)
        self.apply_gradient = apply_gradient
        self.need_err_input = need_err_input
        self.gradient_weights = formats.Vector()
        self.gradient_bias = formats.Vector()

    def initialize(self, device, **kwargs):
        super(GradientDescentBase, self).initialize(device=device, **kwargs)
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

    def generate_data_for_slave(self, slave=None):
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
        if self.gradient_weights.v is not None:
            self.gradient_weights.map_invalidate()
            self.gradient_weights.v[:] = 0
        if self.gradient_bias.v is not None:
            self.gradient_bias.map_invalidate()
            self.gradient_bias.v[:] = 0

    def generate_data_for_master(self):
        if (not self.run_was_called or
                (self.gradient_weights.v is None and
                    self.gradient_bias.v is None)):
            return None
        self.run_was_called = False
        self.gradient_weights.map_read()
        self.gradient_bias.map_read()
        return (self.gradient_weights.v, self.gradient_bias.v)

    def apply_data_from_slave(self, data, slave=None):
        if self.weights.v is not None:
            self.weights.map_write()
            self.weights.v += data[0]
        if self.bias.v is not None:
            self.bias.map_write()
            self.bias.v += data[1]


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

    def export(self, filename):
        """Exports workflow for use on DTV.
        """
        # create temporary folder
        tmppath = os.path.join(config.root.common.cache_dir, "saver_tmp")
        if not os.path.exists(tmppath):
            os.makedirs(tmppath)
        files_to_save = []
        dict_temp = {}
        variables_to_save = []
        # Go through units & save numpy array to binary file
        units_to_export = [self.loader]
        units_to_export.extend(self.fwds)
        for i in range(len(units_to_export)):
            u = units_to_export[i]
            if u.exports is None:
                self.debug("%s continue" % u.__class__.__name__)
                continue
            variables = u.__getstate__()
            for key in variables:
                if key in u.exports:
                    self.debug("%s in attributes to export" % (key))
                    # Save numpy array to binary file
                    if type(getattr(u, key)) == formats.Vector and i >= 1:
                        for j in range(len(getattr(u, key).v.shape)):
                            name = key + "_shape_" + str(j)
                            self.info(name)
                            dict_temp[name] = getattr(u, key).v.shape[j]

                        link_to_numpy = "unit" + str(i - 1) + key + ".bin"

                        dict_temp['link_to_' + key] = link_to_numpy

                        files_to_save.append(
                            self._save_numpy_to_file(
                                getattr(u, key).v, link_to_numpy, tmppath))
                    else:
                        dict_temp[key] = getattr(u, key)
            temp__ = {}
            temp__[u.__class__.__name__] = dict_temp
            variables_to_save.append(temp__)
            dict_temp = {}

        # Save forward elements to yaml.
        yaml_name = 'default.yaml'
        self._save_to_yaml("%s/%s" % (tmppath, yaml_name), variables_to_save)
        # Compress archive
        tar = tarfile.open("%s.tar.gz" % (filename), "w:gz")
        tar.add("%s/%s" % (tmppath, yaml_name),
                arcname=yaml_name, recursive=False)
        for i in range(len(files_to_save)):
            tar.add("%s/%s" % (tmppath, files_to_save[i]),
                    arcname=files_to_save[i], recursive=False)
        tar.close()
        # delete temporary folder
        shutil.rmtree(tmppath)

    def _save_to_yaml(self, yaml_name, to_yaml):
        """Print workflow to yaml-file.
        Parameters:
            yaml_name: filename to save.
        """
        stream = open(yaml_name, "w")
        for i in range(len(to_yaml)):
            yaml.dump(to_yaml[i], stream)
        stream.close()

    def _save_numpy_to_file(self, numpy_vector, numpy_vector_name, path):
        """Save numpy array to binary file.
        Parameters:
            numpy_vector: contains numpy array.
            numpy_vector_name: name of the binary file to save numpy array.
        """
        array_to_save = numpy.float32(numpy_vector.ravel())

        with open("%s/%s" % (path, numpy_vector_name), "wb") as f:
            f.write(array_to_save)
        return numpy_vector_name


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
        None: lambda n, l: tarfile.TarFile.open(n, "wb"),
        "": lambda n, l: tarfile.TarFile.open(n, "wb"),
        "gz": lambda n, l: tarfile.TarFile.gzopen(n, "wb", compresslevel=l),
        "bz2": lambda n, l: tarfile.TarFile.bz2open(n, "wb", compresslevel=l),
        "xz": lambda n, l: tarfile.TarFile.xzopen(n, "wb", preset=l)
    }

    def __init__(self, workflow, **kwargs):
        super(ForwardExporter, self).__init__(workflow, **kwargs)
        self.forwards = []

    def export(self):
        ext = ("." + self.compress) if self.compress else ""
        rel_file_name = "%s_%s_wb.%d.tar%s" % (
            self.prefix, self.suffix, 3 if six.PY3 else 2, ext)
        self.file_name = os.path.join(self.directory, rel_file_name)
        with self._open_file() as tar:
            for index, fwd in enumerate(self.forwards):
                weights, bias = fwd.generate_data_for_slave()
                fileobj = six.BytesIO()
                numpy.savez(fileobj, weights, bias)
                ti = tarfile.TarInfo("%03d_%s.npz" % (index, fwd.name))
                ti.size = fileobj.tell()
                ti.mode = int("666", 8)
                fileobj.seek(0)
                tar.addfile(ti, fileobj=fileobj)
        self.info("Wrote %s" % self.file_name)
        file_name_link = os.path.join(
            self.directory, "%s_current_wb.%d.tar%s" % (
                self.prefix, 3 if six.PY3 else 2, ext))
        if os.path.exists(file_name_link):
            os.remove(file_name_link)
        os.symlink(rel_file_name, file_name_link)

    def _open_file(self):
        return ForwardExporter.CODECS[self.compress](self.file_name,
                                                     self.compress_level)
