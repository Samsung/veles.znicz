"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Oct 29, 2014

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
import numpy.matlib as matlib
from zope.interface import implementer

from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit, ICUDAUnit
from veles.memory import Vector
from veles.mutable import Bool
import veles.prng as prng
from veles.units import IUnit, Unit
from veles.workflow import Repeater, Workflow
from veles.znicz.all2all import All2AllSigmoid
from veles.znicz.evaluator import EvaluatorMSE


class EmptyDeviceMethodsMixin(object):
    def ocl_init(self):
        pass

    def cuda_init(self):
        pass

    def ocl_run(self):
        pass

    def cuda_run(self):
        pass

    def cpu_run(self):
        pass


@implementer(IOpenCLUnit, ICUDAUnit)
class Binarization(AcceleratedUnit, EmptyDeviceMethodsMixin):
    """
    Input Binarization. Input and output is 2d arrays of the same size.
    Each element A(i,j) (in row i and column j) of input is a float
    number between 0 and 1. Each element B(i,j) of output is equal 1 with
    probability A(i,j) and 0 with 1 - A(i,j).
    Must be assigned before initialize():
        input

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: input as batch of samples.
        output: output as batch of samples.
    """
    def __init__(self, workflow, **kwargs):
        super(Binarization, self).__init__(workflow, **kwargs)
        self.output = Vector()
        self.rand = kwargs.get("rand", prng.get())
        self.demand("input", "batch_size")

    def run(self):
        """Batch binarization on CPU only.
        """
        self.output.map_invalidate()
        self.input.map_read()
        self.output.mem[:] = self.input.mem[:]
        self.output.mem[:self.batch_size, :] = self.matlab_binornd(
            1, self.input.mem[:self.batch_size, :])

    def initialize(self, device, **kwargs):
        super(Binarization, self).initialize(device=device, **kwargs)
        if not self.output or self.output.size != self.input.size:
            self.output.reset()
            self.output.mem = numpy.zeros_like(self.input.mem)
        self.output.initialize(self.device)

    def matlab_binornd(self, n, p_in):
        """
        Analogue binornd in Matlab, but n  must be scalar.

        The function generates a matrix of random variables,
        where the element at (i,j) position is generated from binomial
        distribution with the number of trials n and the probability of
        success p_in(i,j).

        Args:
            n (int): number of trials
            p_in (2 dimension numpy.array): success probability matrix
        Returns:
            res (2 dimension numpy.array): matrix of random variables
            generated from the binomial distribution
        """
        p = numpy.copy(p_in)
        if len(p.shape) == 2:
            nrow = p.shape[0]
            ncol = p.shape[1]
            p = numpy.transpose(p)
            p = p.flatten()
            dim = p.shape[0]
            p = matlib.repmat(p, n, 1)
            f = self.rand.rand(n, dim)
            res = f < p
            res = numpy.sum(res, axis=0)
            res = numpy.transpose(res.reshape(ncol, nrow)).reshape(nrow, ncol)
        elif len(p.shape) == 1:
            p = matlib.repmat(p, n, 1)
            dim = p.shape[0]
            p = matlib.repmat(p, n, 1)
            f = self.rand.rand(n, dim)
            res = f < p
            res = numpy.sum(res, axis=0)
        else:  # will make exeption
            raise ValueError("shape of input Binarization class "
                             "must be 1 or 2 dimensions")
        return res


@implementer(IUnit)
class IterationCounter(Unit):
    """
    Simple repeater. Loop is repeated max_iteration iterations
    """
    def __init__(self, workflow, **kwargs):
        """
        Args:
        max_iteration: loop is repeated max_iteration iterations
        """
        super(IterationCounter, self).__init__(workflow, **kwargs)
        self.max_iterations = kwargs["max_iterations"]
        self.iteration = 0
        self.complete = Bool(False)

    def reset(self):
        self.iteration = 0
        self.complete <<= self.iteration > self.max_iterations

    def initialize(self, **kwargs):
        self.complete <<= self.iteration > self.max_iterations

    def run(self):
        self.iteration += 1
        self.complete <<= self.iteration > self.max_iterations


@implementer(IOpenCLUnit, ICUDAUnit)
class BatchWeights(AcceleratedUnit, EmptyDeviceMethodsMixin):
    """Make weigths and biases from batch v and h.
    Must be assigned before initialize():
        v
        h
        batch_size
    Updates after run():
        hbias_batch
        vbias_batch
        W_batch

    Creates within initialize():
        hbias_batch
        vbias_batch
        W_batch

    Attributes:
        v: input data  batch
        h: hidden states of input batch
        batch_size: size of batch
        hbias_batch: bias calculated from h
        vbias_batch: bias calculated from v
        W_batch: weigths calculated from batch v and h
    """
    def __init__(self, workflow, **kwargs):
        super(BatchWeights, self).__init__(workflow, **kwargs)
        self.vbias_batch = Vector()
        self.hbias_batch = Vector()
        self.weights_batch = Vector()
        self.demand("v", "h", "batch_size")

    def initialize(self, device, **kwargs):
        super(BatchWeights, self).initialize(device=device, **kwargs)
        vbias_size = self.v.size // self.v.shape[0]
        hbias_size = self.h.size // self.h.shape[0]
        W_size = vbias_size * hbias_size
        if not self.hbias_batch:
            self.hbias_batch.reset(numpy.zeros((1, hbias_size),
                                               dtype=self.h.mem.dtype))
        else:
            assert self.hbias_batch.size == hbias_size
        if not self.vbias_batch:
            self.vbias_batch.reset(numpy.zeros((1, vbias_size),
                                               dtype=self.h.mem.dtype))
        else:
            assert self.vbias_batch.size == vbias_size
        if not self.weights_batch:
            self.weights_batch.reset(numpy.zeros((vbias_size, hbias_size),
                                                 dtype=self.h.mem.dtype))
        else:
            assert self.weights_batch.size == W_size
        self.init_vectors(self.weights_batch, self.vbias_batch,
                          self.hbias_batch, self.v, self.h)

    def run(self):
        self.v.map_read()
        self.h.map_read()
        for v in self.weights_batch, self.hbias_batch, self.vbias_batch:
            v.map_invalidate()
        self.weights_batch.mem[:] = numpy.dot(
            numpy.transpose(self.v.mem[0: self.batch_size, :]),
            self.h.mem[0: self.batch_size, :]) / \
            self.batch_size
        for bv in (self.vbias_batch, self.v), (self.hbias_batch, self.h):
            bv[0].mem[:] = (numpy.sum(bv[1].mem[:self.batch_size, :], 0) /
                            self.batch_size)
            bv[0].shape = (1, bv[0].size)


class BatchWeights2(BatchWeights):
    """
    Don't remove.
    Dummy class as a workaround for link_attrs behaviour.
    """
    hide_from_registry = True


@implementer(IOpenCLUnit, ICUDAUnit)
class GradientsCalculator(AcceleratedUnit, EmptyDeviceMethodsMixin):
    """
    Making gradients for weights, hbias and vbias, using hbias0, vbias0
    and vbias1, hbias1, which calculated with help BatchWeights.
    Must be assigned before initialize():
        hbias0
        vbias0
        hbias1
        vbias1
        weights1
        weights0

    Updates after run():
        hbias_grad
        vbias_grad
        weights_grad
    Creates within initialize():
        hbias_grad
        vbias_grad
        weights_grad

    Attributes:
        vbias0: calculated with help BatchWeights from v0
        hbias0: calculated with help BatchWeights from h0
        vbias1: calculated with help BatchWeights from v1
        hbias1: calculated with help BatchWeights from h1
        weights1: calculated with help BatchWeights from v1.
        weights0: calculated with help BatchWeights from h1.
        hbias_grad: gradient for hbias
        vbias_grad: gradient for vbias
        weights_grad: gradient for weights
    """
    def __init__(self, workflow, **kwargs):
        super(GradientsCalculator, self).__init__(workflow, **kwargs)
        self.vbias_grad = Vector()
        self.hbias_grad = Vector()
        self.weights_grad = Vector()
        self.demand("hbias1", "vbias1", "hbias0", "vbias0", "weights0",
                    "weights1")

    def initialize(self, device, **kwargs):
        super(GradientsCalculator, self).initialize(device=device, **kwargs)
        if not self.hbias_grad:
            self.hbias_grad.reset(numpy.zeros(self.hbias0.shape,
                                              dtype=self.hbias0.dtype))
        else:
            assert self.hbias_grad.shape == self.hbias0.shape
        if not self.vbias_grad:
            self.vbias_grad.reset(numpy.zeros(self.vbias0.shape,
                                              dtype=self.vbias0.dtype))
        else:
            assert self.vbias_grad.shape == self.vbias0.shape
        if not self.weights_grad:
            self.weights_grad.reset(numpy.zeros(self.weights0.shape,
                                                dtype=self.weights0.dtype))
        else:
            assert self.weights_grad.shape == self.weights0.shape
        for v in (self.weights_grad, self.hbias_grad, self.vbias_grad,
                  self.hbias0, self.vbias0, self.weights0, self.hbias1,
                  self.vbias1, self.weights1):
            v.initialize(self.device)

    def run(self):
        for v in (self.hbias0, self.vbias0, self.weights0,
                  self.hbias1, self.vbias1, self.weights1):
            v.map_read()

        for v in (self.weights_grad, self.vbias_grad, self.hbias_grad):
            v.map_invalidate()

        self.vbias_grad.mem[:] = self.vbias0.mem - self.vbias1.mem
        self.hbias_grad.mem[:] = self.hbias0.mem - self.hbias1.mem
        self.weights_grad.mem[:] = self.weights0.mem - self.weights1.mem


@implementer(IUnit)
class WeightsUpdater(Unit):
    """
    Adds gradiens to weights, bias and hbias
    """
    def __init__(self, workflow, **kwargs):
        super(WeightsUpdater, self).__init__(workflow, **kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.demand("hbias_grad", "vbias_grad", "weights_grad",
                    "weights", "hbias", "vbias")

    def initialize(self, **kwargs):
        pass

    def run(self):
        for v in self.hbias_grad, self.vbias_grad, self.weights:
            v.map_read()
        for v in self.weights, self.hbias, self.vbias:
            v.map_write()

        self.weights.mem += self.learning_rate * \
            self.weights_grad.mem.transpose()
        self.hbias.mem += self.learning_rate * self.hbias_grad.mem.reshape(
            self.hbias.shape)
        self.vbias.mem += self.learning_rate * self.vbias_grad.mem.reshape(
            self.vbias.shape)


@implementer(IOpenCLUnit, ICUDAUnit)
class MemCpy(AcceleratedUnit):
    def __init__(self, workflow, **kwargs):
        super(MemCpy, self).__init__(workflow, **kwargs)
        self.output = Vector()
        self.demand("input")

    def initialize(self, device, **kwargs):
        super(MemCpy, self).initialize(device, **kwargs)
        if (self.output.mem is None or
                self.output.mem.size != self.input.mem.size):
            self.output.reset()
            self.output.mem = numpy.zeros(self.input.mem.shape,
                                          dtype=self.input.mem.dtype)
        self.input.initialize(self.device)
        self.output.initialize(self.device)

    def cuda_init(self):
        pass

    def ocl_init(self):
        pass

    def _gpu_run(self):
        self.input.unmap()
        self.output.unmap()

    def ocl_run(self):
        self._gpu_run()
        self.device.queue_.copy_buffer(self.input.devmem, self.output.devmem,
                                       0, 0, self.input.nbytes)

    def cuda_run(self):
        self._gpu_run()
        self.output.devmem.from_device_async(self.input.devmem)

    def cpu_run(self):
        self.input.map_read()
        self.output.map_invalidate()
        numpy.copyto(self.output.mem, self.input.mem)


class All2AllSigmoidH(All2AllSigmoid):
    """
    Don't remove.
    Dummy class as a workaround for link_attrs behaviour.
    """
    MAPPING = set()
    hide_from_registry = True


class All2AllSigmoidV(All2AllSigmoid):
    """
    Don't remove.
    Dummy class as a workaround for link_attrs behaviour.
    """
    MAPPING = set()
    hide_from_registry = True


class BinarizationGradH(Binarization):
    """
    Don't remove.
    Dummy class as a workaround for link_attrs behaviour.
    """
    hide_from_registry = True


class BinarizationGradV(Binarization):
    """
    Don't remove.
    Dummy class as a workaround for link_attrs behaviour.
    """
    hide_from_registry = True


class GradientRBM(Workflow):
    """This unit produces update weights using minibatch according to the
    algorithm described in
    http://deeplearning.net/tutorial/rbm.html (25.11.14).
    Does Gibbs sampling
    cd_k: number of iterations of Gibbs sampling
    """
    def __init__(self, workflow, **kwargs):
        super(GradientRBM, self).__init__(workflow, **kwargs)
        self.stddev = kwargs["stddev"]
        self.batch_size = -1
        self.mem_cpy = MemCpy(self)
        self.mem_cpy.link_from(self.start_point)
        self.repeater = Repeater(self)
        self.repeater.link_from(self.mem_cpy)
        self.decision = IterationCounter(
            self, max_iterations=kwargs["cd_k"])
        self.decision.link_from(self.repeater)
        self.bino_h = BinarizationGradH(
            self, rand=kwargs.get("rand_h", prng.get()))
        self.bino_h.link_attrs(self.mem_cpy, ("input", "output"))
        self.bino_h.link_from(self.decision)
        self.bino_h.gate_block = self.decision.complete
        self.make_v = All2AllSigmoidV(
            self, weights_stddev=self.stddev, weights_transposed=True,
            output_sample_shape=kwargs["v_size"])
        self.make_v.link_from(self.bino_h)
        self.make_v.link_attrs(self.bino_h, ("input", "output"))
        self.bino_v = BinarizationGradV(
            self, rand=kwargs.get("rand_v", prng.get()))

        self.bino_v.link_attrs(self.make_v, ("input", "output"))
        self.bino_v.link_from(self.make_v)
        self.make_h = All2AllSigmoidH(
            self, weights_stddev=self.stddev,
            output_sample_shape=kwargs["h_size"])
        self.make_h.link_attrs(self.bino_v, ("input", "output"))
        self.make_h.output = self.mem_cpy.output
        self.make_h.link_from(self.bino_v)

        self.repeater.link_from(self.make_h)
        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete
        self.bino_h.gate_block = self.decision.complete

        self.mem_cpy.link_attrs(self, "input")
        self.bino_h.link_attrs(self, "batch_size")
        self.bino_v.link_attrs(self, "batch_size")
        self.make_v.link_attrs(self, "weights")
        self.make_v.link_attrs(self, ("bias", "vbias"))
        self.make_h.link_attrs(self, "weights")
        self.make_h.link_attrs(self, ("bias", "hbias"))
        self.link_attrs(self.make_h, "output")
        self.link_attrs(self.bino_v, ("v1", "output"))
        self.link_attrs(self.make_h, ("h1", "output"))
        self.demand("input", "weights", "hbias", "vbias", "batch_size")

    def run(self):
        self.decision.reset()
        super(GradientRBM, self).run()


class All2AllSigmoidWithForeignWeights(All2AllSigmoid):
    """
    Dummy class as a workaround for link_attrs behavior.
    """
    MAPPING = set()
    hide_from_registry = True


class BinarizationEval(Binarization):
    """
    Dummy class as a workaround for link_attrs behavior.
    """
    hide_from_registry = True


class EvaluatorRBM(Workflow):
    def __init__(self, workflow, **kwargs):
        super(EvaluatorRBM, self).__init__(workflow, **kwargs)
        self.run_is_blocking = True
        self.binarization = BinarizationEval(
            self, rand=kwargs.get("rand", prng.get()))
        self.binarization.link_from(self.start_point)
        self.rec = All2AllSigmoidWithForeignWeights(
            self, output_sample_shape=kwargs["bias_shape"],
            weights_transposed=True)
        self.rec.link_from(self.binarization)
        self.rec.link_attrs(self.binarization, ("input", "output"))
        self.mse = EvaluatorMSE(self, squared_mse=True,
                                error_function_averaged=False)
        self.mse.link_from(self.rec)
        self.mse.link_attrs(self.rec, "output")
        self.mse.link_attrs(self.rec, ("output", "output"))
        self.end_point.link_from(self.mse)

        self.binarization.link_attrs(self, "input", "batch_size")
        self.rec.link_attrs(self, "weights")
        self.mse.link_attrs(self, "target", "batch_size")
        self.link_attrs(self.rec, ("vbias", "bias"))
        self.demand("input", "weights", "target")

    @property
    def output(self):
        return self.vbias
