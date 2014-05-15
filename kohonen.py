"""
Created on May 05, 2014

Kohonen units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy

import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units
import veles.error as error


class Kohonen(nn_units.Forward):
    """Kohonen forward layer.

    Should be assigned before initialize():
        input

    Updates after run():
        output

    Creates within initialize():
        weights
        output

    Attributes:
        input: input as batch of samples.
        output: output as batch of samples.
        shape: shape of the output layer (may be Vector).
        weights_transposed: assume weights matrix as a transposed one.
        weights_filling: rand weight filling
                         ("uniform" (default) or "gaussian")
        weights_stddev: magnitude of uniform weight distribution.
    """
    def __init__(self, workflow, **kwargs):
        super(Kohonen, self).__init__(workflow, **kwargs)
        self._shape = kwargs["shape"]
        self.weights_filling = kwargs.get("weights_filling", "uniform")
        self.weights_stddev = kwargs.get("weights_stddev", None)
        self.input = None

    def init_unpickled(self):
        super(Kohonen, self).init_unpickled()
        self.cl_sources_["kohonen.cl"] = {}

    @property
    def shape(self):
        return self._shape

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

        output_size = int(numpy.prod(self._shape))

        if self.weights_stddev is None:
            # Get weights magnitude and cap it to 0.05
            self.weights_stddev = min(self.get_weights_magnitude(), 0.05)
        n_weights = (self.input.v.size // self.input.v.shape[0] * output_size)
        if self.weights.v is None or self.weights.v.size != n_weights:
            self.weights.reset()
            self.weights.v = numpy.zeros(n_weights, dtype=self.input.v.dtype)
            if self.weights_filling == "uniform":
                self.rand.fill(self.weights.v, -self.weights_stddev,
                               self.weights_stddev)
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

        if self.device is None:
            return

        defines = {
            'BLOCK_SIZE': self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)],
            'BATCH': self.output.v.shape[0],
            'SAMPLE_LENGTH': self.weights.v.size // output_size,
            'NEURONS_NUMBER': output_size}
        if self.weights_transposed:
            defines['WEIGHTS_TRANSPOSED'] = 1
        self.build_program(defines, "kohonen_%d_%d.cl" %
                           (self.input.v.size // self.input.v.shape[0],
                            output_size),
                           dtype=self.input.v.dtype)

        self.assign_kernel("feed_layer")
        self.set_args(self.input, self.weights, self.output)

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

        self.execute_kernel(self._global_size_, self._local_size_).wait()

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
        input
        weights
        batch_size

    Updates after run():
        weights

    Attributes:
        weights: weights of the current layer.
        input: input of the current layer as batch of 1D samples.
        krn_dist_: computes distances between input and neuron weights.
        krn_argmin_: finds indexes of minimal computed distances.
        krn_gravity_: computes gravity to the winner neuron.
        krn_compute_gradients_: computes gradient for weights.
        krn_apply_gradients_: applies gradient to weights.
    """
    def __init__(self, workflow, **kwargs):
        super(KohonenTrain, self).__init__(workflow, **kwargs)
        self.distances = formats.Vector()
        self.argmin = formats.Vector()
        self.coords = formats.Vector()
        self.input = None
        self.time = 0
        self._sigma = 0
        self.gradient_decay = kwargs.get("gradient_decay",
                                         lambda t: 0.1 / (1.0 + t * 0.05))
        self.radius_decay = kwargs.get("radius_decay",
                                       lambda t: 1.0 / (1.0 + t * 0.05))

    def init_unpickled(self):
        super(KohonenTrain, self).init_unpickled()
        self.cl_sources_["kohonen.cl"] = {"TRAIN": 1}
        self.krn_distance_ = None
        self.krn_argmin_ = None
        self.krn_gravity_ = None
        self.krn_compute_gradients_ = None
        self.krn_apply_gradients_ = None
        numpy_version = [int(v) for v in numpy.__version__.split('.')]
        if numpy_version[0] == 1 and numpy_version[1] < 8:
            self._numpy_linalg_norm = self._numpy_legacy_linalg_norm
        else:
            self._numpy_linalg_norm = self._numpy_1_8_linalg_norm

    @property
    def gravity_radius(self):
        return self.radius_decay(self.time) * self._sigma

    @property
    def gradient_multiplier(self):
        return self.gradient_decay(self.time)

    def initialize(self, device, **kwargs):
        super(KohonenTrain, self).initialize(device=device, **kwargs)

        batch_size = self.input.v.shape[0]
        if self.weights_transposed:
            self._sample_length = self.weights.v.shape[0]
            self._neurons_number = self.weights.v.shape[1]
        else:
            self._sample_length = self.weights.v.shape[1]
            self._neurons_number = self.weights.v.shape[0]

        self.distances.reset()
        self.distances.v = numpy.zeros(
            [batch_size, self._neurons_number],
            dtype=self.weights.v.dtype)

        self.argmin.reset()
        self.argmin.v = numpy.zeros(batch_size, dtype=numpy.int32)

        self.coords.reset()
        self.coords.v = numpy.zeros([self._neurons_number, 2],
                                    dtype=self.weights.v.dtype)
        sz = self._neurons_number
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

        self._sigma = (self.coords.v.ravel().max() -
                       self.coords.v.ravel().min()) * 1.42
        self.weights.initialize(self.device)
        self.input.initialize(self.device)
        self.distances.initialize(self.device)
        self.argmin.initialize(self.device)
        self.coords.initialize(self.device)

        if self.device is None:
            return

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.weights.v.dtype)]
        chunk_size = self._neurons_number // self.device.max_group_size
        if chunk_size < 2:
            chunk_size = self._neurons_number // 2 + 1
        self.chunked_group_size = int(numpy.ceil(self._neurons_number /
                                                 chunk_size))

        defines = {
            'BLOCK_SIZE': block_size,
            'BATCH': batch_size,
            'SAMPLE_LENGTH': self._sample_length,
            'NEURONS_NUMBER': self._neurons_number,
            'CHUNK_SIZE': chunk_size,
            'coord_type':  "%s%d" %
            (opencl_types.numpy_dtype_to_opencl(self.coords.v.dtype),
             self.coords.v.shape[-1])
        }
        if self.weights_transposed:
            defines['WEIGHTS_TRANSPOSED'] = 1
        self.build_program(defines, "kohonen_train_%d_%d.cl" %
                           (self._sample_length, self._neurons_number),
                           dtype=self.weights.v.dtype)

        self.ocl_consts_ = numpy.zeros(1, dtype=self.weights.v.dtype)

        self.krn_distance_ = self.get_kernel("compute_distance")
        self.krn_distance_.set_args(self.input.v_, self.weights.v_,
                                    self.distances.v_)

        self.krn_argmin_ = self.get_kernel("compute_argmin")
        self.krn_argmin_.set_args(self.distances.v_, self.argmin.v_)

        self.krn_gravity_ = self.get_kernel("compute_gravity")
        self.krn_gravity_.set_args(self.argmin.v_, self.coords.v_)
        self.krn_gravity_.set_arg(3, self.distances.v_)

        self.krn_apply_gradient_ = self.get_kernel("apply_gradient")
        self.krn_apply_gradient_.set_args(self.input.v_, self.distances.v_)
        self.krn_apply_gradient_.set_arg(3, self.weights.v_)

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.weights.v.dtype)]

        self._gs_distance = [
            formats.roundup(self._sample_length, block_size),
            formats.roundup(self._neurons_number, block_size)]
        self._ls_distance = [block_size, block_size]

    def iteration(fn):
        def wrapped(self, *args, **kwargs):
            self.input.unmap()
            self.weights.unmap()
            self.distances.unmap()
            self.argmin.unmap()
            self.coords.unmap()
            result = fn(self, *args, **kwargs)
            self.time += 1
            return result
        return wrapped

    def _numpy_1_8_linalg_norm(self, dist):
        return numpy.linalg.norm(dist, axis=1)

    def _numpy_legacy_linalg_norm(self, dist):
        return [numpy.linalg.norm(dist[i]) for i in range(dist.shape[0])]

    @iteration
    def cpu_run(self):
        """Does Kohonen's learning iteration on CPU.
        """
        batch_size = self.input.v.shape[0]
        neurons_number = self._neurons_number
        dists = numpy.empty(neurons_number)
        gradients = numpy.zeros(self.weights.v.shape)
        sigma = self.gravity_radius
        gmult = self.gradient_multiplier
        for sindex in range(batch_size):
            dist = self.weights.v - self.input[sindex]
            winner = numpy.argmin(self._numpy_linalg_norm(dist))
            winner_coords = self.coords.v[winner]
            for nindex in range(neurons_number):
                dist = self.coords.v[nindex] - winner_coords
                dists[nindex] = numpy.sum(dist * dist)
            gravity = numpy.exp(dists / (-2 * sigma * sigma))
            gradients += gravity.reshape((1, neurons_number)).transpose() * \
                (self.input[sindex] - self.weights.v) * gmult
        self.weights.v += gradients

    @iteration
    def ocl_run(self):
        """Does Kohonen's learning iteration using OpenCL.
        """
        batch_size = self.input.v.shape[0]

        self.execute_kernel(self._gs_distance, self._ls_distance,
                            self.krn_distance_).wait()
        self.execute_kernel([self.chunked_group_size],
                            [self.chunked_group_size], self.krn_argmin_).wait()
        self.ocl_consts_[0] = self.gravity_radius
        self.krn_gravity_.set_arg(2, self.ocl_consts_[0:1])
        self.execute_kernel([batch_size, self._neurons_number], None,
                            self.krn_gravity_).wait()
        self.ocl_consts_[0] = self.gradient_multiplier
        self.krn_apply_gradient_.set_arg(2, self.ocl_consts_[0:1])
        self.execute_kernel([self.chunked_group_size], None,
                            self.krn_apply_gradient_).wait()

    iteration = staticmethod(iteration)
