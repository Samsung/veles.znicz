"""
Created on May 05, 2014

Kohonen units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division
import numpy
import opencl4py as cl
from zope.interface import implementer

import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.opencl_units import IOpenCLUnit, OpenCLUnit
import veles.znicz.decision as decision
import veles.random_generator as prng
from veles.znicz.decision import TrivialDecision


class KohonenBase(object):
    """Common base of Kohonen units.
    """

    def init_unpickled(self):
        super(KohonenBase, self).init_unpickled()
        numpy_version = [int(mem) for mem in numpy.__version__.split('.')]
        if numpy_version[0] == 1 and numpy_version[1] < 8:
            self.numpy_linalg_norm = self._numpy_legacy_linalg_norm
        else:
            self.numpy_linalg_norm = self._numpy_1_8_linalg_norm

    def _numpy_1_8_linalg_norm(self, dist):
        return numpy.linalg.norm(dist, axis=1)

    def _numpy_legacy_linalg_norm(self, dist):
        return [numpy.linalg.norm(dist[i]) for i in range(dist.shape[0])]


@implementer(IOpenCLUnit)
class KohonenForward(KohonenBase, OpenCLUnit):
    """Kohonen forward layer.

    Must be assigned before initialize():
        input
        weights
        minibatch_offset (if total == True)
        minibatch_size (if total == True)
        batch_size (if total == True)
        argmins speeds up run() if linked from KohonenTrainer

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: input as batch of samples.
        weights: the weights of the neurons in Kohonen layer.
        output: the list of winners.
        total: if total=True is passed in __init__(), the overall winners table
    """
    def __init__(self, workflow, **kwargs):
        super(KohonenForward, self).__init__(workflow, **kwargs)
        self.demand("input", "weights")
        self.argmins = None
        self._distances = formats.Vector()
        self.output = formats.Vector()
        self._batch_size = 0
        self._chunk_size_ = 0
        self.weights_transposed = False
        self.total = formats.Vector() if kwargs.get("total", False) else None
        if self.total is not None:
            self.minibatch_offset = None
            self.minibatch_size = None
            self.batch_size = None

    def init_unpickled(self):
        super(KohonenForward, self).init_unpickled()
        self.cl_sources_["kohonen.cl"] = {"FORWARD": 1}

    @property
    def neurons_number(self):
        return self.weights.mem.shape[0]

    @property
    def sample_length(self):
        return self.weights.mem.shape[1]

    @property
    def chunk_size(self):
        return self._chunk_size_

    def initialize(self, device, **kwargs):
        super(KohonenForward, self).initialize(device=device, **kwargs)

        assert self.input.mem.shape[1] == self.sample_length
        batch_size = self.input.mem.shape[0]

        self.output.reset()
        self.output.mem = numpy.zeros(batch_size, dtype=numpy.int32)
        if self.argmins is None:
            self._distances.reset()
            self._distances.mem = numpy.zeros(
                [batch_size, self.neurons_number],
                dtype=self.weights.mem.dtype)

        if self.total is not None:
            self.total.reset()
            self.total.mem = numpy.zeros(self.batch_size, dtype=numpy.int32)
            self._minibatch_offset_ = numpy.zeros(1, dtype=numpy.int32)

        if self.device is None:
            return

        self.output.initialize(self.device)
        if self.argmins is None:
            self.input.initialize(self.device)
            self.weights.initialize(self.device)
            self._distances.initialize(device)
        elif self.total is None:
            return
        if self.total is not None:
            self.total.initialize(self.device)

        copy_chunk_size = int(numpy.ceil(batch_size /
                                         self.device.max_group_size))
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.weights.mem.dtype)]
        chunk_size = self.neurons_number // self.device.max_group_size
        if chunk_size < 2:
            chunk_size = self.neurons_number // 2 + 1
        self.argmin_group_size = \
            int(numpy.ceil(self.neurons_number / chunk_size))
        defines = {
            'BLOCK_SIZE': block_size,
            'BATCH': batch_size,
            'SAMPLE_LENGTH': self.sample_length,
            'NEURONS_NUMBER': self.neurons_number,
            'CHUNK_SIZE': chunk_size,
            'COPY_CHUNK_SIZE': copy_chunk_size,
        }
        if self.weights_transposed:
            defines['WEIGHTS_TRANSPOSED'] = 1
        self.build_program(defines, "kohonen_forward_%d_%d_%d.cl" %
                           (batch_size, self.sample_length,
                            self.neurons_number),
                           dtype=self.weights.mem.dtype)

        if self.total is not None:
            self._set_total_global_size_ = \
                [int(numpy.ceil(batch_size / copy_chunk_size))]
            self._krn_set_total_ = self.get_kernel("set_total")
            self._krn_set_total_.set_args(self.output.devmem, cl.skip,
                                          self.total.devmem)
        if self.argmins is not None:
            return

        self._krn_distances_ = self.get_kernel("calculate_distances")
        self._krn_distances_.set_args(self.input.devmem, self.weights.devmem,
                                      self._distances.devmem)

        self._krn_argmin_ = self.get_kernel("calculate_argmin")
        self._krn_argmin_.set_args(self._distances.devmem, self.output.devmem,
                                   None)

        self._gs_distance = [
            formats.roundup(self.neurons_number, block_size),
            formats.roundup(batch_size, block_size)]
        self._ls_distance = [block_size, block_size]

    def ocl_run(self):
        self.output.unmap()
        if self.total is not None:
            self.total.unmap()

        if self.argmins is None:
            self.input.unmap()
            self.weights.unmap()
            self.execute_kernel(self._gs_distance, self._ls_distance,
                                self._krn_distances_)
            self.execute_kernel([self.argmin_group_size],
                                [self.argmin_group_size],
                                self._krn_argmin_)
        else:
            self.argmins.unmap()
            self.argmins.map_read()
            self.output.map_write()
            self.output.mem[:] = self.argmins.mem
            self.output.unmap()
            self.argmins.unmap()

        if self.total is not None:
            self._minibatch_offset_[0] = \
                self.minibatch_offset - self.minibatch_size
            self._krn_set_total_.set_arg(1, self._minibatch_offset_)
            self.execute_kernel(self._set_total_global_size_, None,
                                self._krn_set_total_)

    def cpu_run(self):
        self.output.map_invalidate()

        if self.argmins is not None:
            self.argmins.map_read()
            self.output.mem[:] = self.argmins.mem
        else:
            self.input.map_read()
            self.weights.map_read()

        if self.total is not None:
            self.total.map_invalidate()

        length = self.minibatch_size if self.total is not None \
            else self.input.mem.shape[0]
        for sindex in range(length):
            if self.argmins is None:
                dist = self.weights.mem - self.input[sindex]
                winner = numpy.argmin(self.numpy_linalg_norm(dist))
                self.output[sindex] = winner
            else:
                winner = self.argmins[sindex]
            if self.total is not None:
                index = sindex + self.minibatch_offset - self.minibatch_size
                self.total[index] = winner


@implementer(IOpenCLUnit)
class KohonenTrainer(KohonenBase, OpenCLUnit):
    """KohonenForward train pass.

    Must be assigned before initialize():
        input
        shape

    Creates within initialize():
        weights
        winners
        argmins
        _distances
        _coords

    Updates after run():
        weights

    Attributes:
        weights: weights of the current layer.
        input: input of the current layer as batch of 1D samples.
        krn_dist_: computes distances between input and neuron weights.
        _krn_argmin_: finds indexes of minimal computed distances.
        krn_gravity_: computes gravity to the winner neuron.
        krn_apply_gradients_: applies gradient to weights.
    """
    def __init__(self, workflow, shape, **kwargs):
        super(KohonenTrainer, self).__init__(workflow, **kwargs)
        self.demand("input", "shape")
        self._distances = formats.Vector()
        self.argmins = formats.Vector()
        self._coords = formats.Vector()
        self.weights = formats.Vector()
        self.winners = formats.Vector()
        self._shape = shape
        self.weights_filling = kwargs.get("weights_filling", "uniform")
        self.weights_stddev = kwargs.get("weights_stddev", None)
        self.weights_transposed = kwargs.get("weights_transposed", False)
        self.time = 0
        self._sigma = 0
        self.gradient_decay = kwargs.get("gradient_decay",
                                         lambda t: 0.1 / (1.0 + t * 0.05))
        self.radius_decay = kwargs.get("radius_decay",
                                       lambda t: 1.0 / (1.0 + t * 0.05))

    def init_unpickled(self):
        super(KohonenTrainer, self).init_unpickled()
        self.cl_sources_["kohonen.cl"] = {"TRAIN": 1}
        self._krn_distances_ = None
        self._krn_argmin_ = None
        self._krn_gravity_ = None
        self._krn_compute_gradients_ = None
        self._krn_apply_gradients_ = None

    @property
    def gravity_radius(self):
        return self.radius_decay(self.time) * self._sigma

    @property
    def gradient_multiplier(self):
        return self.gradient_decay(self.time)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    def initialize(self, device, **kwargs):
        super(KohonenTrainer, self).initialize(device=device, **kwargs)

        self._neurons_number = self.shape[0] * self.shape[1]
        self._sample_length = self.input.mem.size // self.input.mem.shape[0]

        # Initialize weights
        if self.weights_stddev is None:
            # Get weights magnitude and cap it to 0.05
            self.weights_stddev = min(self._get_weights_magnitude(), 0.05)
        weights_size = (self._sample_length * self._neurons_number)
        if self.weights.mem is None:
            self.weights.reset()
            self.weights.mem = numpy.zeros(weights_size,
                                           dtype=self.input.mem.dtype)
            filling = {
                "uniform": lambda rand: rand.fill(
                    self.weights.mem, -self.weights_stddev,
                    self.weights_stddev),
                "gaussian": lambda rand: rand.fill_normal_real(
                    self.weights.mem, 0, self.weights_stddev)
            }
            filling[self.weights_filling](prng.get())
            self.weights.mem = self.weights.mem.reshape((
                self._neurons_number, self._sample_length))
        if self.weights_transposed:
            # Reshape weights as a matrix:
            wtrncopy = self.weights.mem.transpose().copy()
            self.weights.mem.shape = wtrncopy.shape
            self.weights.mem[:] = wtrncopy[:]
        self._sample_length = \
            self.weights.mem.shape[0 if self.weights_transposed else 1]

        # Initialize winners
        self.winners.reset()
        self.winners.mem = numpy.zeros(self._neurons_number, dtype=numpy.int32)

        # Initialize distances
        batch_size = self.input.mem.shape[0]
        self._distances.reset()
        self._distances.mem = numpy.zeros(
            [batch_size, self._neurons_number],
            dtype=self.weights.mem.dtype)

        self.argmins.reset()
        self.argmins.mem = numpy.zeros(batch_size, dtype=numpy.int32)

        self._coords.reset()
        self._coords.mem = numpy.zeros([self._neurons_number, 2],
                                       dtype=self.weights.mem.dtype)
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
        mem = self._coords.mem
        for _row in range(rows):
            x = x_min + (x_step * 0.5 if _row & 1 else 0)
            for _col in range(cols):
                mem[offs, 0] = x
                mem[offs, 1] = y
                offs += 1
                x += x_step
            y += y_step

        self._sigma = (self._coords.mem.ravel().max() -
                       self._coords.mem.ravel().min()) * 1.42

        if self.device is None:
            return
        self.input.initialize(self.device)
        self.weights.initialize(self.device)
        self.winners.initialize(self.device)
        self.argmins.initialize(self.device)
        self._distances.initialize(self.device)
        self._coords.initialize(self.device)

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.weights.mem.dtype)]
        chunk_size = self._neurons_number // self.device.max_group_size
        if chunk_size < 2:
            chunk_size = self._neurons_number // 2 + 1
        self.argmin_group_size = int(numpy.ceil(float(self._neurons_number) /
                                                chunk_size))
        defines = {
            'BLOCK_SIZE': block_size,
            'BATCH': batch_size,
            'SAMPLE_LENGTH': self._sample_length,
            'NEURONS_NUMBER': self._neurons_number,
            'CHUNK_SIZE': chunk_size,
            'GRADIENT_CHUNK_SIZE': self.device.max_group_size,
            'coord_type':  "%s%d" %
            (opencl_types.numpy_dtype_to_opencl(self._coords.mem.dtype),
             self._coords.mem.shape[-1])
        }
        if self.weights_transposed:
            defines['WEIGHTS_TRANSPOSED'] = 1
        self.build_program(defines, "kohonen_train_%d_%d_%d.cl" %
                           (batch_size, self._sample_length,
                            self._neurons_number),
                           dtype=self.weights.mem.dtype)

        self.ocl_consts_ = numpy.zeros(1, dtype=self.weights.mem.dtype)

        self._krn_distances_ = self.get_kernel("calculate_distances")
        self._krn_distances_.set_args(self.input.devmem, self.weights.devmem,
                                      self._distances.devmem)

        self._krn_argmin_ = self.get_kernel("calculate_argmin")
        self._krn_argmin_.set_args(self._distances.devmem, self.argmins.devmem,
                                   self.winners.devmem)

        self._krn_gravity_ = self.get_kernel("compute_gravity")
        self._krn_gravity_.set_args(self.argmins.devmem, self._coords.devmem)
        self._krn_gravity_.set_arg(3, self._distances.devmem)

        self._krn_apply_gradient_ = self.get_kernel("apply_gradient")
        self._krn_apply_gradient_.set_args(self.input.devmem,
                                           self._distances.devmem)
        self._krn_apply_gradient_.set_arg(3, self.weights.devmem)

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.weights.mem.dtype)]

        self._gs_distance = [
            formats.roundup(self._neurons_number, block_size),
            formats.roundup(batch_size, block_size)]
        self._ls_distance = [block_size, block_size]

    def iteration(fn):
        def wrapped(self, *args, **kwargs):
            result = fn(self, *args, **kwargs)
            self.time += 1
            return result
        return wrapped

    @iteration
    def cpu_run(self):
        batch_size = self.input.mem.shape[0]
        neurons_number = self._neurons_number
        dists = numpy.empty(neurons_number)
        gradients = numpy.zeros(self.weights.mem.shape)
        sigma = self.gravity_radius
        gmult = self.gradient_multiplier
        self.input.map_read()
        self.weights.map_invalidate()
        self.winners.map_invalidate()

        for sindex in range(batch_size):
            dist = self.weights.mem - self.input[sindex]
            winner = numpy.argmin(self.numpy_linalg_norm(dist))
            self.winners[winner] += 1
            winner_coords = self._coords.mem[winner]
            for nindex in range(neurons_number):
                dist = self._coords.mem[nindex] - winner_coords
                dists[nindex] = numpy.sum(dist * dist)
            gravity = numpy.exp(dists / (-2 * sigma * sigma))
            gradients += gravity.reshape((1, neurons_number)).transpose() * \
                (self.input[sindex] - self.weights.mem) * gmult
        self.weights.mem += gradients

    @iteration
    def ocl_run(self):
        self.input.unmap()
        self.weights.unmap()
        self.winners.unmap()
        self._distances.unmap()
        self.argmins.unmap()
        self._coords.unmap()

        batch_size = self.input.mem.shape[0]
        self.execute_kernel(self._gs_distance, self._ls_distance,
                            self._krn_distances_)
        self.execute_kernel([self.argmin_group_size],
                            [self.argmin_group_size],
                            self._krn_argmin_)
        self.ocl_consts_[0] = self.gravity_radius
        self._krn_gravity_.set_arg(2, self.ocl_consts_[0:1])
        self.execute_kernel([batch_size, self._neurons_number], None,
                            self._krn_gravity_)
        self.ocl_consts_[0] = self.gradient_multiplier
        self._krn_apply_gradient_.set_arg(2, self.ocl_consts_[0:1])
        self.execute_kernel(
            [int(numpy.ceil(self._sample_length / self.device.max_group_size)),
             self.device.max_group_size],
            None, self._krn_apply_gradient_)

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
        if self.input.mem.dtype in (numpy.complex64, numpy.complex128):
            return 1.0 / d
        return 9.0 / d


class KohonenDecision(TrivialDecision, decision.DecisionBase):
    def __init__(self, workflow, **kwargs):
        super(KohonenDecision, self).__init__(workflow, **kwargs)
        self.weights_copy = formats.Vector()
        self.winners_copy = formats.Vector()

    def on_training_finished(self):
        """This method is supposed to be overriden in inherited classes.
        """
        self.weights.map_read()
        self.winners.map_write()
        self.weights_copy.mem = numpy.copy(self.weights.mem)
        self.winners_copy.mem = numpy.copy(self.winners.mem)
        self.winners.mem[:] = 0
