"""
Created on May 05, 2014

Kohonen units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy

import veles.config as config
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units
import veles.error as error
from veles.config import root


class Kohonen(nn_units.Forward):
    """Kohonen.

    Should be assigned before initialize():
        input

    Updates after run():
        output

    Creates within initialize():
        weights
        bias
        output

    Attributes:
        input: input as batch of samples.
        output: output as batch of samples.
        weights: matrix of weights.
        bias: bias.
        output_shape: shape of the output layer (may be Vector).
        krn_: OpenCL kernel.
        s_activation: activation define for OpenCL source.
        weights_transposed: assume weights matrix as a transposed one.

        weights_filling: rand weight filling
                         ("uniform" (default) or "gaussian")
        weights_magnitude: magnitude of uniform weight distribution.
        weights_stddev: StdDev of normal weight distributtion
    """
    def __init__(self, workflow, **kwargs):
        output_shape = kwargs.get("output_shape")
        kwargs["output_shape"] = output_shape

        weights_filling = kwargs.get("weights_filling", "uniform")
        weights_magnitude = kwargs.get("weights_magnitude", None)
        weights_stddev = kwargs.get("weights_stddev", 0.05)

        kwargs["weights_magnitude"] = weights_magnitude
        kwargs["weights_filling"] = weights_filling
        kwargs["weights_stddev"] = weights_stddev

        super(Kohonen, self).__init__(workflow, **kwargs)
        self.input = None
        self.weights_filling = weights_filling
        self.weights_stddev = weights_stddev
        self.weights_magnitude = weights_magnitude
        self.output_shape = output_shape

    def init_unpickled(self):
        super(Kohonen, self).init_unpickled()
        self.cl_sources_["kohonen.cl"] = {}
        self.krn_ = None

    def get_weights_magnitude(self):
        """
        Returns: weights magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.

        Doen't matter for classic kohonen networks,
        got values as in All2AllTanh.
        """
        if self.input.v.dtype in (numpy.complex64, numpy.complex128):
            return (1.0 / self.input.supposed_maxvle /
                    (self.input.v.size // self.input.v.shape[0]))
        return (9.0 / self.input.supposed_maxvle /
                (self.input.v.size // self.input.v.shape[0]))

    def initialize(self, device, **kwargs):
        super(Kohonen, self).initialize(device=device, **kwargs)

        output_size = int(numpy.prod(self.output_shape))

        if self.weights_magnitude is None:
            # Get weights magnitude and cap it to 0.05
            self.weights_magnitude = min(self.get_weights_magnitude(), 0.05)
        n_weights = (self.input.v.size // self.input.v.shape[0] * output_size)
        if self.weights.v is None or self.weights.v.size != n_weights:
            self.weights.reset()
            self.weights.v = numpy.zeros(n_weights, dtype=self.input.v.dtype)
            if self.weights_filling == "uniform":
                self.rand.fill(self.weights.v, -self.weights_magnitude,
                               self.weights_magnitude)
            elif self.weights_filling == "gaussian":
                self.rand.fill_normal_real(self.weights.v, 0,
                                           self.weights_stddev)
            else:
                raise error.ErrBadFormat(
                    "Unknown weights_filling = %s encountered" %
                    self.weights_filling)
            self.weights.v = self.weights.v.reshape([
                output_size, self.input.v.size // self.input.v.shape[0]])
            # Reshape weights as a matrix:
            if self.weights_transposed:
                a = self.weights.v.transpose().copy()
                self.weights.v.shape = a.shape
                self.weights.v[:] = a[:]

        if (self.output.v is None or
                self.output.v.size != self.input.v.shape[0] * output_size):
            self.output.reset()
            self.output.v = numpy.zeros([self.input.v.shape[0], output_size],
                                        dtype=self.input.v.dtype)

        self.input.initialize(self.device)
        self.output.initialize(self.device)
        self.weights.initialize(self.device)
        self.bias.initialize(self.device)

        if self.device is None:
            return

        if self.krn_ is None:
            defines = {
                'BLOCK_SIZE': self.device.device_info.BLOCK_SIZE[
                    opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)],
                'H': self.weights.v.size // output_size,
                'Y': output_size,
                'BATCH': self.output.v.shape[0]}
            if self.weights_transposed:
                defines['WEIGHTS_TRANSPOSED'] = 1
            self.build_program(defines, "%s/kohonen_%d_%d.cl" %
                               (config.root.common.cache_dir,
                                self.input.v.size // self.input.v.shape[0],
                                output_size),
                               dtype=self.input.v.dtype)

            self.krn_ = self.get_kernel("feed_layer")
            self.krn_.set_arg(0, self.input.v_)
            self.krn_.set_arg(1, self.weights.v_)
            self.krn_.set_arg(2, self.output.v_)

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)]
        self._global_size_ = [formats.roundup(output_size, block_size),
                              formats.roundup(self.output.v.shape[0],
                                              block_size)]
        self._local_size_ = [block_size, block_size]

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        self.output.unmap()
        self.input.unmap()
        self.weights.unmap()

        self.execute_kernel(self.krn_, self._global_size_,
                            self._local_size_).wait()

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.output.map_invalidate()
        self.input.map_read()
        self.weights.map_read()
        a = formats.reshape(
            self.input.v, [self.input.v.shape[0],
                           self.input.v.size // self.input.v.shape[0]])
        b = self.weights.v
        if not self.weights_transposed:
            b = b.transpose()
        v = numpy.dot(a, b)
        self.output.v[:] = v[:]


class KohonenTrain(nn_units.GradientDescentBase):
    """Kohonen train pass.

    Should be assigned before initialize():
        h
        weights
        batch_size

    Updates after run():
        weights

    Attributes:
        weights: weights of the current layer.
        h: input of the current layer as batch of 1D samples.
        krn_dist_: computes distances between input and neuron weights.
        krn_argmin_: finds indexes of minimal computed distances.
        krn_gravity_: computes gravity to the winner neuron.
        krn_compute_gradients_: computes gradient for weights.
        krn_apply_gradients_: applies gradient to weights.
    """
    def __init__(self, workflow, **kwargs):
        super(KohonenTrain, self).__init__(workflow, **kwargs)
        self.reduce_size = 64
        self.distance = formats.Vector()
        self.argmin = formats.Vector()
        self.coords = formats.Vector()
        self.sigma = None

    def init_unpickled(self):
        super(KohonenTrain, self).init_unpickled()
        self.cl_sources_["kohonen.cl"] = {}
        self.krn_distance_ = None
        self.krn_argmin_ = None
        self.krn_gravity_ = None
        self.krn_compute_gradients_ = None
        self.krn_apply_gradients_ = None

    def initialize(self, device, **kwargs):
        super(KohonenTrain, self).initialize(device=device, **kwargs)

        if self.weights_transposed:
            self._input_size = self.weights.v.shape[0]
            self._output_size = self.weights.v.shape[1]
        else:
            self._input_size = self.weights.v.shape[1]
            self._output_size = self.weights.v.shape[0]

        if (self.gradient_weights.v is None or
                self.gradient_weights.v.size != self.weights.v.size):
            self.gradient_weights.reset()
            self.gradient_weights.v = numpy.zeros_like(self.weights.v)

        if (self.distance.v is None or
                self.distance.v.size != self._output_size):
            self.distance.reset()
            self.distance.v = numpy.zeros(
                [self.h.v.shape[0], self._output_size],
                dtype=self.weights.v.dtype)

        if (self.argmin.v is None or
                self.argmin.v.size != self._output_size):
            self.argmin.reset()
            self.argmin.v = numpy.zeros(self._output_size, dtype=numpy.int32)

        if (self.coords.v is None or
                self.coords.v.size != self._output_size):
            self.coords.reset()
            self.coords.v = numpy.zeros([self._output_size, 2],
                                        dtype=self.weights.v.dtype)
            sz = self._output_size
            rows = int(numpy.round(numpy.sqrt(sz)))
            cols = sz // rows
            if sz % rows != 0:
                cols += 1
            x_min = -1.0
            x_max = 1.0
            y_min = -1.0
            y_max = 1.0
            x_step = (x_max - x_min) / (cols - 1) if cols > 1 else 0
            y = y_min
            y_step = (y_max - y_min) / (rows - 1) if rows > 1 else 0
            offs = 0
            v = self.coords.v
            for _row in range(rows):
                x = x_min + (x_step * 0.5 if _row & 1 else 0)
                for _col in range(cols):
                    v[offs, 0] = x
                    v[offs, 1] = y
                    offs += 1
                    x += x_step
                y += y_step
        if self.sigma is None:
            self.sigma = (self.coords.v.ravel().max() -
                          self.coords.v.ravel().min()) * 1.42

        self.weights.initialize(self.device)
        self.h.initialize(self.device)
        self.distance.initialize(self.device)
        self.argmin.initialize(self.device)
        self.coords.initialize(self.device)
        self.gradient_weights.initialize(self.device)

        if self.device is None:
            return

        if self.program_ is None:
            block_size = self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(self.weights.v.dtype)]
            self.reduce_size = min(self.reduce_size, self._output_size)

            defines = {
                'BLOCK_SIZE': block_size,
                'BATCH': self.h.v.shape[0],
                'H': self._input_size,
                'Y': self._output_size,
                'REDUCE_SIZE': self.reduce_size,
                'coord_type':  "%s%d" %
                (opencl_types.numpy_dtype_to_opencl(self.coords.v.dtype),
                 self.coords.v.shape[-1])
            }
            if self.weights_transposed:
                defines['WEIGHTS_TRANSPOSED'] = 1
            self.build_program(
                defines, "%s/kohonen_train_%d_%d.cl" % (
                    root.common.cache_dir,
                    self._input_size, self._output_size),
                dtype=self.weights.v.dtype)

            self.krn_const_ = numpy.zeros(2, dtype=self.weights.v.dtype)

            self.krn_distance_ = self.get_kernel("compute_distance")
            self.krn_distance_.set_arg(0, self.h.v_)
            self.krn_distance_.set_arg(1, self.weights.v_)
            self.krn_distance_.set_arg(2, self.distance.v_)

            self.krn_argmin_ = self.get_kernel("compute_argmin")
            self.krn_argmin_.set_arg(0, self.distance.v_)
            self.krn_argmin_.set_arg(1, self.argmin.v_)

            self.krn_gravity_ = self.get_kernel("compute_gravity")
            self.krn_gravity_.set_arg(0, self.argmin.v_)
            self.krn_gravity_.set_arg(1, self.coords.v_)
            self.krn_gravity_.set_arg(2, self.distance.v_)

            self.krn_compute_gradient_ = self.get_kernel("compute_gradient")
            self.krn_compute_gradient_.set_arg(0, self.h.v_)
            self.krn_compute_gradient_.set_arg(1, self.weights.v_)
            self.krn_compute_gradient_.set_arg(2, self.gradient_weights.v_)

            self.krn_apply_gradient_ = self.get_kernel("apply_gradient")
            self.krn_apply_gradient_.set_arg(0, self.gradient_weights.v_)
            self.krn_apply_gradient_.set_arg(1, self.weights.v_)
            self.krn_apply_gradient_.set_arg(2, self.distance.v_)

            block_size = self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(self.weights.v.dtype)]

            self._gs_distance = [
                formats.roundup(self._input_size, block_size),
                formats.roundup(self._output_size, block_size)]
            self._ls_distance = [block_size, block_size]

            if self.weights_transposed:
                self._gs_compute_gradient = [
                    formats.roundup(self._output_size, block_size),
                    formats.roundup(self.h.v.shape[0], block_size)]
            else:
                self._gs_compute_gradient = [
                    formats.roundup(self.h.v.shape[0], block_size),
                    formats.roundup(self._output_size, block_size)]
            self._ls_compute_gradient = [block_size, block_size]

    def cpu_run(self):
        """Do gradient descent.
        """
        raise error.ErrNotImplemented()

    def ocl_run(self):
        """Do gradient descent.
        """
        self.h.unmap()
        self.weights.unmap()
        self.distance.unmap()
        self.argmin.unmap()
        self.gradient_weights.unmap()
        self.coords.unmap()

        batch_size = self.batch_size or self.h.v.shape[0]

        self.execute_kernel(self.krn_distance_, self._gs_distance,
                            self._ls_distance).wait()
        self.execute_kernel(self.krn_argmin_, [self.reduce_size * batch_size],
                            [self.reduce_size]).wait()

        self.krn_const_[0] = self.sigma
        self.krn_gravity_.set_arg(3, self.krn_const_[0:1])
        self.execute_kernel(self.krn_gravity_, [batch_size, self._output_size],
                            None).wait()

        self.execute_kernel(self.krn_compute_gradient_,
                            self._gs_compute_gradient,
                            self._ls_compute_gradient).wait()

        self.krn_const_[0] = -self.learning_rate / batch_size
        self.krn_const_[1] = -self.learning_rate * self.weights_decay
        self.krn_apply_gradient_.set_arg(3, self.krn_const_[0:1])
        self.krn_apply_gradient_.set_arg(4, self.krn_const_[1:2])
        self.execute_kernel(self.krn_apply_gradient_, [self.weights.v.size],
                            None).wait()
