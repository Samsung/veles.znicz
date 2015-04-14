# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 05, 2014

Kohonen units.

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


from __future__ import division
import numpy
import opencl4py as cl
from zope.interface import implementer

from veles.units import Unit, IUnit
import veles.memory as formats
import veles.opencl_types as opencl_types
from veles.accelerated_units import IOpenCLUnit, AcceleratedUnit, INumpyUnit
import veles.prng as prng
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


@implementer(IOpenCLUnit, INumpyUnit)
class KohonenForward(KohonenBase, AcceleratedUnit):
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
        self._chunk_size_ = 0
        self.weights_transposed = False
        self.total = formats.Vector() if kwargs.get("total", False) else None
        if self.total is not None:
            self.minibatch_offset = None
            self.minibatch_size = None
            self.batch_size = None

    def init_unpickled(self):
        super(KohonenForward, self).init_unpickled()
        self.sources_["kohonen"] = {"FORWARD": 1}

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

        self.output.reset(numpy.zeros(batch_size, dtype=numpy.int32))
        if self.argmins is None:
            self._distances.reset(numpy.zeros(
                [batch_size, self.neurons_number],
                dtype=self.weights.mem.dtype))

        if self.total is not None:
            self.total.reset(numpy.zeros(self.batch_size, dtype=numpy.int32))
            self._minibatch_offset_ = numpy.zeros(1, dtype=numpy.int32)

    def ocl_init(self):
        batch_size = self.input.mem.shape[0]
        self.output.initialize(self.device)
        if self.argmins is None:
            self.input.initialize(self.device)
            self.weights.initialize(self.device)
            self._distances.initialize(self.device)
        elif self.total is None:
            return
        if self.total is not None:
            self.total.initialize(self.device)

        copy_chunk_size = int(numpy.ceil(batch_size /
                                         self.device.max_group_size))
        chunk_size = self.neurons_number // self.device.max_group_size
        if chunk_size < 2:
            chunk_size = self.neurons_number // 2 + 1
        self.argmin_group_size = \
            int(numpy.ceil(self.neurons_number / chunk_size))

        block_size = self.device.device_info.get_block_size(
            kernel="matrix_multiplication", dtype=self.input.dtype)

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
        self.build_program(defines, "%s_%d_%d_%d" %
                           (self.__class__.__name__,
                            batch_size, self.sample_length,
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

    def numpy_run(self):
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


@implementer(IOpenCLUnit, INumpyUnit)
class KohonenTrainer(KohonenBase, AcceleratedUnit):
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
    def __init__(self, workflow, **kwargs):
        super(KohonenTrainer, self).__init__(workflow, **kwargs)
        self._distances = formats.Vector()
        self.argmins = formats.Vector()
        self._coords = formats.Vector()
        self.weights = formats.Vector()
        self.winners = formats.Vector()
        self.weights_filling = kwargs.get("weights_filling", "uniform")
        self.weights_stddev = kwargs.get("weights_stddev", None)
        self.weights_transposed = kwargs.get("weights_transposed", False)
        self.time = 0
        self._sigma = 0
        self.gradient_decay = kwargs.get("gradient_decay",
                                         lambda t: 0.1 / (1.0 + t * 0.05))
        self.radius_decay = kwargs.get("radius_decay",
                                       lambda t: 1.0 / (1.0 + t * 0.05))
        self.demand("input", "shape")
        self._shape = kwargs.get("shape")

    def init_unpickled(self):
        super(KohonenTrainer, self).init_unpickled()
        self.sources_["kohonen"] = {"TRAIN": 1}
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
        if not self.weights:
            self.weights.reset(numpy.zeros(weights_size,
                                           dtype=self.input.mem.dtype))
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
        else:
            assert self.weights.shape == (self._neurons_number,
                                          self._sample_length)
        if self.weights_transposed:
            # Reshape weights as a matrix:
            wtrncopy = self.weights.mem.transpose().copy()
            self.weights.mem.shape = wtrncopy.shape
            self.weights.mem[:] = wtrncopy[:]
        self._sample_length = \
            self.weights.mem.shape[0 if self.weights_transposed else 1]

        # Initialize winners
        self.winners.reset(numpy.zeros(self._neurons_number, numpy.int32))

        # Initialize distances
        batch_size = self.input.mem.shape[0]
        self._distances.reset(numpy.zeros(
            [batch_size, self._neurons_number],
            dtype=self.weights.mem.dtype))
        self.argmins.reset(numpy.zeros(batch_size, dtype=numpy.int32))
        self._coords.reset(numpy.zeros([self._neurons_number, 2],
                                       dtype=self.weights.mem.dtype))
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

    def ocl_init(self):
        self.input.initialize(self.device)
        self.weights.initialize(self.device)
        self.winners.initialize(self.device)
        self.argmins.initialize(self.device)
        self._distances.initialize(self.device)
        self._coords.initialize(self.device)

        batch_size = self.input.mem.shape[0]
        chunk_size = self._neurons_number // self.device.max_group_size
        if chunk_size < 2:
            chunk_size = self._neurons_number // 2 + 1
        self.argmin_group_size = int(numpy.ceil(float(self._neurons_number) /
                                                chunk_size))

        block_size = self.device.device_info.get_block_size(
            kernel="matrix_multiplication", dtype=self.input.dtype)

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
        self.build_program(defines, "%s_%d_%d_%d" %
                           (self.__class__.__name__,
                            batch_size, self._sample_length,
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

        self._gs_distance = [
            formats.roundup(self._neurons_number, block_size),
            formats.roundup(batch_size, block_size)]
        self._ls_distance = [block_size, block_size]

    def iteration(fn):
        def wrapped(self, *args, **kwargs):
            result = fn(self, *args, **kwargs)
            self.time += 1
            return result

        name = getattr(fn, '__name__', getattr(fn, 'func', wrapped).__name__)
        wrapped.__name__ = name + '_iteration'
        return wrapped

    @iteration
    def numpy_run(self):
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
        self.unmap_vectors(self.input, self.weights, self.winners,
                           self._distances, self.argmins, self._coords)

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
        d = self.input.max_supposed * self._sample_length
        if self.input.mem.dtype in (numpy.complex64, numpy.complex128):
            return 1.0 / d
        return 9.0 / d


class KohonenDecision(TrivialDecision):
    """
    Stops Kohonen network training on the incremental weights difference basis.

    Attributes:
        weights_mem: the neurons' weights, copied from "weights.mem".
        winners_mem: the winning neurons, copied from "winners.mem".
        weights_diff: the difference between previous and current weights.
    """
    def __init__(self, workflow, **kwargs):
        super(KohonenDecision, self).__init__(workflow, **kwargs)
        self.weights_mem = numpy.empty(shape=(0, 0), dtype=numpy.float32)
        self._prev_weights = numpy.empty(shape=(0, 0), dtype=numpy.float32)
        self.winners_mem = numpy.empty(shape=(0, 0))
        self._previous_weights = None
        self.weights_min_diff = kwargs.get("weights_min_diff", 0)
        self.demand("weights", "winners")

    @property
    def weights_diff(self):
        if self.weights_mem.size * self._prev_weights.size == 0:
            return numpy.inf
        return numpy.linalg.norm(self.weights_mem - self._prev_weights)

    def on_training_finished(self):
        """This method is supposed to be overriden in inherited classes.
        """
        self.weights.map_read()
        self.winners.map_invalidate()

        self._prev_weights = self.weights_mem.copy()
        if self.weights_mem.shape != self.weights.mem.shape:
            self.weights_mem.resize(self.weights.mem.shape, refcheck=False)
        numpy.copyto(self.weights_mem, self.weights.mem)
        if self.winners_mem.shape != self.winners.mem.shape:
            self.winners_mem.resize(self.winners.mem.shape, refcheck=False)
        numpy.copyto(self.winners_mem, self.winners.mem)
        self.winners.mem[:] = 0

    def train_improve_condition(self):
        if self.weights_diff < self.weights_min_diff:
            return True
        return super(KohonenDecision, self).train_improve_condition()

    def fill_statistics(self, stats):
        stats.append("weights diff: %f" % self.weights_diff)


@implementer(IUnit)
class KohonenValidator(Unit):
    """
    Maps the winning Kohonen neurons with real categories.

    It accumulates winners from "input" attribute which should be connected to
    KohonenForward's "output" and learns categories from "samples_by_label".
    samples_by_label must be label indices for each sample (that is, a list).

    Attributes:
        result: the resulting mapping between Kohonen neurons and real
                categories.
        fitness: the ratio of samples classified right to the overall number.
    """
    def __init__(self, workflow, **kwargs):
        super(KohonenValidator, self).__init__(workflow, **kwargs)
        self.demand("input", "minibatch_indices", "minibatch_size",
                    "samples_by_label", "shape")
        self.accumulated_input = []
        self._fitness = 0
        self._fitness_by_label = []
        self._fitness_by_neuron = []
        self._result = []
        self._need_validate = False

    def initialize(self, **kwargs):
        self.accumulated_input.clear()
        self.accumulated_input.extend([
            set() for _ in range(self.neurons_count)])
        self._fitness = 0
        self._result.clear()
        self._result.extend([set() for _ in range(len(self.samples_by_label))])
        self._fitness_by_label.extend([
            0 for _ in range(len(self.samples_by_label))])
        self._fitness_by_neuron.extend([0 for _ in range(self.neurons_count)])
        self._overall = sum([len(m) for m in self.samples_by_label])
        assert self._overall > 0
        assert self.neurons_count >= len(self.samples_by_label)
        self._need_validate = True

    def reset(self):
        for acc in self.accumulated_input:
            acc.clear()
        self._need_validate = True

    def run(self):
        self.input.map_read()
        self.minibatch_indices.map_read()

        for i in range(self.minibatch_size):
            self.accumulated_input[self.input[i]].add(
                self.minibatch_indices[i])
        self._need_validate = True

    @property
    def neurons_count(self):
        return self.shape[0] * self.shape[1]

    @property
    def result(self):
        self._validate()
        return self._result

    @property
    def fitness(self):
        self._validate()
        return self._fitness

    @property
    def fitness_by_label(self):
        self._validate()
        return self._fitness_by_label

    @property
    def fitness_by_neuron(self):
        self._validate()
        return self._fitness_by_neuron

    def _validate(self):
        """
        We have the matrix of intersection sizes, rows represent neurons and
        columns represent labels. The problem is to take the numbers from our
        matrix so that the sum is maximal and there are no numbers on the same
        row.
        The algorithm is to first take the maximal number from matrix, then
        the most significant one which stands on a different row, and repeat
        the previous step until the work is done.
        The difficulty is N*L log(N*L).
        """
        if not self._need_validate:
            return
        intersections = []
        for neuron in range(self.neurons_count):
            for label, members in enumerate(self.samples_by_label):
                intersections.append((
                    len(self.accumulated_input[neuron].intersection(members)),
                    neuron, label))
        intersections.sort(reverse=True)
        self._result.clear()
        self._result.extend([set() for _ in range(len(self.samples_by_label))])
        fitted = 0
        fitted_by_label = [0 for _ in range(len(self.samples_by_label))]
        fitted_by_neuron = [0 for _ in range(self.neurons_count)]
        pos = 0
        banned_neurons = set()
        while (intersections[pos][0] > 0 and
               len(banned_neurons) < self.neurons_count):
            while (intersections[pos][1] in banned_neurons):
                pos += 1
            fit, neuron, label = intersections[pos]
            fitted += fit
            fitted_by_label[label] += fit
            fitted_by_neuron[neuron] = fit
            self._result[label].add(neuron)
            banned_neurons.add(neuron)
        self._fitness = fitted / self._overall
        for label, members in enumerate(self.samples_by_label):
            self._fitness_by_label[label] = fitted_by_label[label] / \
                len(members)
        for neuron, wins in enumerate(self.accumulated_input):
            self._fitness_by_neuron[neuron] = \
                fitted_by_neuron[neuron] / len(wins) if len(wins) > 0 else 0
        self.reset()
        self._need_validate = False
        self.info("Fitness: %.2f", self._fitness)
        self.info("Neurons mapping: %s", dict(enumerate(self._result)))
