"""
Created on May 05, 2014

Kohonen units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy

import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.nn_units as nn_units
import veles.znicz.decision as decision
import veles.random_generator as rnd


class KohonenForward(nn_units.Forward):
    """Kohonen forward layer.

    Must be assigned before initialize():
        input
        weights
        shape

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: input as batch of samples.
        weights: the weights of the neurons in Kohonen layer.
        output: output as batch of samples.
        shape: shape of the output layer (may be Vector).
        weights_transposed: assume weights matrix as a transposed one.
        weights_filling: rand weight filling
                         ("uniform" (default) or "gaussian")
        weights_stddev: magnitude of uniform weight distribution.
    """
    def __init__(self, workflow, **kwargs):
        super(KohonenForward, self).__init__(workflow, **kwargs)
        self.input = None
        self.weights = None
        self.shape = None
        self.output = formats.Vector()

    def init_unpickled(self):
        super(KohonenForward, self).init_unpickled()
        self.cl_sources_["kohonen.cl"] = {}

    def initialize(self, device, **kwargs):
        super(KohonenForward, self).initialize(device=device, **kwargs)

        neurons_number = self.shape[0] * self.shape[1]
        self.output.reset()
        self.output.v = numpy.zeros((self.input.v.shape[0], neurons_number),
                                    dtype=self.input.v.dtype)

        if self.device is None:
            return

        self.input.initialize(self.device)
        self.weights.initialize(self.device)
        self.output.initialize(self.device)

        defines = {
            'BLOCK_SIZE': self.device.device_info.BLOCK_SIZE[
                opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)],
            'BATCH': self.output.v.shape[0],
            'SAMPLE_LENGTH': self.weights.v.size // neurons_number,
            'NEURONS_NUMBER': neurons_number}
        if self.weights_transposed:
            defines['WEIGHTS_TRANSPOSED'] = 1
        self.build_program(defines, "kohonen_%d_%d.cl" %
                           (self.input.v.size // self.input.v.shape[0],
                            neurons_number),
                           dtype=self.input.v.dtype)

        self.assign_kernel("feed_layer")
        self.set_args(self.input, self.weights, self.output)

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.input.v.dtype)]
        self._global_size_ = [formats.roundup(neurons_number, block_size),
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


class KohonenTrainer(nn_units.GradientDescentBase):
    """KohonenForward train pass.

    Must be assigned before initialize():
        input
        shape
        epoch_ended

    Creates within initialize():
        weights
        winners
        _distances
        _argmin
        _coords

    Updates after run():
        weights

    Attributes:
        weights: weights of the current layer.
        input: input of the current layer as batch of 1D samples.
        krn_dist_: computes distances between input and neuron weights.
        krn_argmin_: finds indexes of minimal computed distances.
        krn_gravity_: computes gravity to the winner neuron.
        krn_apply_gradients_: applies gradient to weights.
    """
    def __init__(self, workflow, shape, **kwargs):
        super(KohonenTrainer, self).__init__(workflow, **kwargs)
        self._distances = formats.Vector()
        self._argmin = formats.Vector()
        self._coords = formats.Vector()
        self.weights = formats.Vector()
        self.winners = formats.Vector()
        self._shape = shape
        self.weights_filling = kwargs.get("weights_filling", "uniform")
        self.weights_stddev = kwargs.get("weights_stddev", None)
        self.weights_transposed = kwargs.get("weights_transposed", False)
        self.input = None
        self.epoch_ended = None
        self.time = 0
        self._sigma = 0
        self.gradient_decay = kwargs.get("gradient_decay",
                                         lambda t: 0.1 / (1.0 + t * 0.05))
        self.radius_decay = kwargs.get("radius_decay",
                                       lambda t: 1.0 / (1.0 + t * 0.05))

    def init_unpickled(self):
        super(KohonenTrainer, self).init_unpickled()
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

    @property
    def shape(self):
        return self._shape

    def initialize(self, device, **kwargs):
        super(KohonenTrainer, self).initialize(device=device, **kwargs)

        self._neurons_number = self.shape[0] * self.shape[1]
        self._sample_length = self.input.v.size // self.input.v.shape[0]

        # Initialize weights
        if self.weights_stddev is None:
            # Get weights magnitude and cap it to 0.05
            self.weights_stddev = min(self._get_weights_magnitude(), 0.05)
        weights_size = (self._sample_length * self._neurons_number)
        if self.weights.v is None:
            self.weights.reset()
            self.weights.v = numpy.zeros(weights_size,
                                         dtype=self.input.v.dtype)
            filling = {
                "uniform": lambda rand: rand.fill(
                    self.weights.v, -self.weights_stddev, self.weights_stddev),
                "gaussian": lambda rand: rand.fill_normal_real(
                    self.weights.v, 0, self.weights_stddev)
            }
            filling[self.weights_filling](rnd.get())
            self.weights.v = self.weights.v.reshape((
                self._neurons_number, self._sample_length))
        if self.weights_transposed:
            # Reshape weights as a matrix:
            wtrncopy = self.weights.v.transpose().copy()
            self.weights.v.shape = wtrncopy.shape
            self.weights.v[:] = wtrncopy[:]
        self._sample_length = self.weights.v.shape[0 if self.weights_transposed
                                                   else 1]

        # Initialize winners
        self.winners.reset()
        self.winners.v = numpy.zeros(self._neurons_number, dtype=numpy.int32)

        # Initialize distances
        batch_size = self.input.v.shape[0]
        self._distances.reset()
        self._distances.v = numpy.zeros(
            [batch_size, self._neurons_number],
            dtype=self.weights.v.dtype)

        self._argmin.reset()
        self._argmin.v = numpy.zeros(batch_size, dtype=numpy.int32)

        self._coords.reset()
        self._coords.v = numpy.zeros([self._neurons_number, 2],
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
        v = self._coords.v
        for _row in range(rows):
            x = x_min + (x_step * 0.5 if _row & 1 else 0)
            for _col in range(cols):
                v[offs, 0] = x
                v[offs, 1] = y
                offs += 1
                x += x_step
            y += y_step

        self._sigma = (self._coords.v.ravel().max() -
                       self._coords.v.ravel().min()) * 1.42

        if self.device is None:
            return
        self.input.initialize(self.device)
        self.weights.initialize(self.device)
        self.winners.initialize(self.device)
        self._distances.initialize(self.device)
        self._argmin.initialize(self.device)
        self._coords.initialize(self.device)

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
            (opencl_types.numpy_dtype_to_opencl(self._coords.v.dtype),
             self._coords.v.shape[-1])
        }
        if self.weights_transposed:
            defines['WEIGHTS_TRANSPOSED'] = 1
        self.build_program(defines, "kohonen_train_%d_%d.cl" %
                           (self._sample_length, self._neurons_number),
                           dtype=self.weights.v.dtype)

        self.ocl_consts_ = numpy.zeros(1, dtype=self.weights.v.dtype)

        self.krn_distance_ = self.get_kernel("compute_distance")
        self.krn_distance_.set_args(self.input.v_, self.weights.v_,
                                    self._distances.v_)

        self.krn_argmin_ = self.get_kernel("compute_argmin")
        self.krn_argmin_.set_args(self._distances.v_, self._argmin.v_,
                                  self.winners.v_)

        self.krn_gravity_ = self.get_kernel("compute_gravity")
        self.krn_gravity_.set_args(self._argmin.v_, self._coords.v_)
        self.krn_gravity_.set_arg(3, self._distances.v_)

        self.krn_apply_gradient_ = self.get_kernel("apply_gradient")
        self.krn_apply_gradient_.set_args(self.input.v_, self._distances.v_)
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
            self._distances.unmap()
            self._argmin.unmap()
            self._coords.unmap()
            result = fn(self, *args, **kwargs)
            self.time += 1
            return result
        return wrapped

    @iteration
    def cpu_run(self):
        """Does KohonenForward's learning iteration on CPU.
        """
        batch_size = self.input.v.shape[0]
        neurons_number = self._neurons_number
        dists = numpy.empty(neurons_number)
        gradients = numpy.zeros(self.weights.v.shape)
        sigma = self.gravity_radius
        gmult = self.gradient_multiplier

        if self.epoch_ended:
            self.winners.v[:] = 0

        for sindex in range(batch_size):
            dist = self.weights.v - self.input[sindex]
            winner = numpy.argmin(self._numpy_linalg_norm(dist))
            self.winners[winner] += 1
            winner_coords = self._coords.v[winner]
            for nindex in range(neurons_number):
                dist = self._coords.v[nindex] - winner_coords
                dists[nindex] = numpy.sum(dist * dist)
            gravity = numpy.exp(dists / (-2 * sigma * sigma))
            gradients += gravity.reshape((1, neurons_number)).transpose() * \
                (self.input[sindex] - self.weights.v) * gmult
        self.weights.v += gradients

    @iteration
    def ocl_run(self):
        """Does KohonenForward's learning iteration using OpenCL.
        """
        batch_size = self.input.v.shape[0]
        if self.epoch_ended:
            self.winners.map_write()
            self.winners.v[:] = 0
            self.winners.unmap()

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

    def _get_weights_magnitude(self):
        """
        Returns: weights magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.

        Doesn't matter for classic Kohonen networks,
        get values as in All2AllTanh.
        """
        d = self.input.supposed_maxvle * self._sample_length
        if self.input.v.dtype in (numpy.complex64, numpy.complex128):
            return 1.0 / d
        return 9.0 / d

    def _numpy_1_8_linalg_norm(self, dist):
        return numpy.linalg.norm(dist, axis=1)

    def _numpy_legacy_linalg_norm(self, dist):
        return [numpy.linalg.norm(dist[i]) for i in range(dist.shape[0])]


class KohonenDecision(decision.DecisionBase):
    def on_training_finished(self):
        """This method is supposed to be overriden in inherited classes.
        """
        self.winners.map_read()
        self.weights.map_read()
        self.winners.unmap()
        self.weights.unmap()
