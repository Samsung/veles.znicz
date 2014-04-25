"""
Created on Jan 28, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import shutil
import tarfile
import yaml

import veles.config as config
import veles.formats as formats
from veles.opencl_units import OpenCLUnit, OpenCLWorkflow
import veles.rnd as rnd
from veles.units import Repeater


class Forward(OpenCLUnit):
    """Base class for forward propagation units.

    Attributes:
        input: input layer values.
        output: output layer values.
        weights: weights.
        bias: bias.
        weights_magnitude: magnitude of the random distribution of weights.
        rand: rnd.Rand() object for initial weights generation.
    """
    def __init__(self, workflow, **kwargs):
        weights_magnitude = kwargs.get("weights_magnitude")
        rand = kwargs.get("rand", rnd.default)
        weights_transposed = kwargs.get("weights_transposed", False)
        kwargs["weights_magnitude"] = weights_magnitude
        kwargs["rand"] = rand
        kwargs["weights_transposed"] = weights_transposed
        kwargs["view_group"] = kwargs.get("view_group", "WORKER")
        super(Forward, self).__init__(workflow, **kwargs)
        self.input = None
        self.output = formats.Vector()
        self.weights = formats.Vector()
        self.bias = formats.Vector()
        self.weights_magnitude = weights_magnitude
        self.rand = rand
        self.weights_transposed = weights_transposed
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


class GD(OpenCLUnit):
    """Base class for gradient descent units.

    Attributes:
        h: input layer values.
        y: output layer values.
        err_y: error to backpropagate.
        err_h: backpropagated error.
        weights: weights.
        bias: bias.
        batch_size: current minibatch size.
        global_alpha: gradient descent speed (positive).
        global_lambda: coefficient (positive or zero) for weights
                       regularization term (lambda/2 * sum(weights^2)).
        batch_size: effective batch size (if None, get it from y).
        weights_transposed: assume weights matrix as a transposed one.
        store_gradient: will save gradient as separate Vector().
        apply_gradient: will apply gradient.
    """
    def __init__(self, workflow, **kwargs):
        global_alpha = kwargs.get("global_alpha", 0.01)
        global_lambda = kwargs.get("global_lambda", 0.00005)
        weights_transposed = kwargs.get("weights_transposed", False)
        store_gradient = kwargs.get("store_gradient", workflow.is_slave)
        apply_gradient = kwargs.get("apply_gradient", not workflow.is_slave)
        kwargs["global_alpha"] = global_alpha
        kwargs["global_lambda"] = global_lambda
        kwargs["weights_transposed"] = weights_transposed
        kwargs["store_gradient"] = store_gradient
        kwargs["apply_gradient"] = apply_gradient
        kwargs["view_group"] = kwargs.get("view_group", "TRAINER")
        super(GD, self).__init__(workflow, **kwargs)
        self.h = None
        self.y = None
        self.err_y = None  # formats.Vector()
        self.err_h = formats.Vector()
        self.weights = None
        self.bias = None
        self.def_attr("batch_size", None)
        self.global_alpha = global_alpha
        self.global_lambda = global_lambda
        self.weights_transposed = weights_transposed
        self.store_gradient = store_gradient
        self.apply_gradient = apply_gradient
        self.gradient_weights = formats.Vector()
        self.gradient_bias = formats.Vector()

    def generate_data_for_slave(self, slave=None):
        return (self.global_alpha, self.global_lambda)

    def apply_data_from_master(self, data):
        self.global_alpha = data[0]
        self.global_lambda = data[1]
        if self.gradient_weights.v is None or self.gradient_bias.v is None:
            return
        self.gradient_weights.map_invalidate()
        self.gradient_weights.v[:] = 0
        self.gradient_bias.map_invalidate()
        self.gradient_bias.v[:] = 0

    def generate_data_for_master(self):
        if (not self.run_was_called or
                self.gradient_weights.v is None or
                self.gradient_bias.v is None):
            return None
        self.run_was_called = False
        self.gradient_weights.map_read()
        self.gradient_bias.map_read()
        return (self.gradient_weights.v, self.gradient_bias.v)

    def apply_data_from_slave(self, data, slave=None):
        self.weights.map_write()
        self.bias.map_write()
        self.weights.v += data[0]
        self.bias.v += data[1]


class NNWorkflow(OpenCLWorkflow):
    """Base class for neural network workflows.

    Attributes:
        repeater: Repeater unit.
        loader: loader unit.
        forward: list of the forward units.
        ev: evaluator unit.
        decision: decision unit.
        gd: list of the gradient descent units.
    """
    def __init__(self, workflow, **kwargs):
        super(NNWorkflow, self).__init__(workflow, **kwargs)
        self.repeater = Repeater(self)
        self.loader = None
        self.forward = []
        self.ev = None
        self.decision = None
        self.gd = []
        self.power = None

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
        units_to_export.extend(self.forward)
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
