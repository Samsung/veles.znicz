"""
Created on Oct 29, 2014
Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""
from __future__ import division
import math
import numpy
from zope.interface import implementer

from veles.memory import Vector
from veles.accelerated_units import AcceleratedUnit, IOpenCLUnit
from veles.znicz.all2all import All2All

import numpy.matlib as matlib
import veles.znicz.gd as gd


def matlab_binornd(n, p_in):
    """
    Analogue binornd in Matlab, but n  must be scalar.

    The function generates a matrix of random variables,
    where the element at (i,j) position is generated from binomial
    distribution with the number of trials n and the probability of
    success p_in(i,j).

    Args:
        n (int): number of trials
        p_in (2 or 1 dimension numpy.array): success probability matrix
    Returns:
        res (2 or 1 dimension numpy.array): matrix of random variables
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
        f = numpy.random.random([n, dim])
        res = f < p
        res = numpy.sum(res, axis=0)
        res = numpy.transpose(res.reshape(ncol, nrow)).reshape(nrow, ncol)
    elif len(p.shape) == 1:
        p = matlib.repmat(p, n, 1)
        dim = p.shape[0]
        p = matlib.repmat(p, n, 1)
        f = numpy.random.random([n, dim])
        res = f < p
        res = numpy.sum(res, axis=0)
    else:  # will make exeption
        raise ValueError("p_in has more than two dimension.")
    return(res)


class All2AllRBM(All2All):
    """Input minibatch must containes number between 0  and 1.
    All2All with Sigmoid activation f(x) = 1 / exp(1.0 + exp(-x)),
    where x := binornd(1, input minibatch).
    This unit also creates weight vbias for reconstruction.
    """
    def __init__(self, workflow, **kwargs):
        super(All2AllRBM, self).__init__(workflow, **kwargs)
        self.vbias = Vector()

    def initialize(self, device, **kwargs):
        self.s_activation = "ACTIVATION_Sigmoid"
        super(All2AllRBM, self).initialize(device=device, **kwargs)
        self.output.max_supposed = 10
        self.vbias.mem = numpy.zeros((1, self.input.shape[1]),
                                     dtype=self.input.mem.dtype)
        self.bias.mem.shape = (1, self.bias.mem.shape[0])
        self.vbias.initialize(self.device)
        # TODO(d.podoprikhin) specify type of the numpy.arrays like
        #     vbias vbias weights
        # TODO(d.podoprikhin) make sd = 1 as usert define constant

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.input.map_read()
        v0 = self.input.mem.copy()
        v0 = v0[0:self.batch_size, :]
        v0 = matlab_binornd(1, v0)
        self.input.map_invalidate()
        self.input.mem[0:self.batch_size, :] = v0[:]
        super(All2AllRBM, self).cpu_run()
        self.output.map_write()
        mem = self.output.mem
        # 1 / (1 + numpy.exp(-mem))
        numpy.exp(-mem, mem)
        mem += 1
        numpy.reciprocal(mem, mem)


class GradientDescentRBM(gd.GradientDescent):
    """This unit produces update weights using minibatch according to the
    algorithm described in
    http://deeplearning.net/tutorial/rbm.html (25.11.14).
    """
    def __init__(self, workflow, **kwargs):
        super(GradientDescentRBM, self).__init__(workflow, **kwargs)
        self.cd_k = kwargs.get("cd_k", 1)
        self.cl_sources_["gradient_descent_relu"] = {}

    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        self.err_output.mem *= 1.0 - numpy.exp(-self.output.mem)

    def cpu_run(self):
        """Do gradient descent.
        """
        for v in (self.weights, self.bias, self.vbias, self.input,
                  self.output):
            v.map_read()
        vbias0, hbias0 = (
            numpy.sum(v.mem[0: self.batch_size, :], axis=0) / self.batch_size
            for v in (self.input, self.output))
        W0 = numpy.dot(numpy.transpose(self.input.mem), self.output.mem)
        W0 /= self.batch_size
        h1 = self.output.mem.copy()
        h1 = h1[0:self.batch_size, :]

        for _ in range(self.cd_k):
            h1 = matlab_binornd(1, h1)
            v1 = numpy.dot(h1, self.weights.mem)
            v1 += self.vbias.mem
            v1 = 1 / (1 + numpy.exp(-v1))
            v1 = matlab_binornd(1, v1)
            h1 = numpy.dot(v1, self.weights.mem.transpose())
            h1 += self.bias.mem
            h1 = 1 / (1 + numpy.exp(-h1))
        vbias1, hbias1 = (numpy.sum(v, axis=0) / self.batch_size
                          for v in (v1, h1))
        W1 = numpy.dot(numpy.transpose(v1), h1) / v1.shape[0]
        vbias_grad, hbias_grad = numpy.subtract((vbias0, hbias0),
                                                (vbias1, hbias1))
        vbias_grad.shape = (1, vbias_grad.size)
        hbias_grad.shape = (1, hbias_grad.size)
        W_grad = W0 - W1
        for v in (self.vbias, self.bias, self.weights):
            v.map_write()
        l_rate = 0.001

        self.bias.mem += l_rate * hbias_grad
        self.vbias.mem += l_rate * vbias_grad
        self.weights.mem += l_rate * W_grad.transpose()

    def initialize(self, device, **kwargs):
        super(GradientDescentRBM, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_args(self.err_output.devmem,
                                      self.output.devmem)


@implementer(IOpenCLUnit)
class EvaluatorRBM(AcceleratedUnit):
    """Evaluates the quality of the autoencoder.
    v := binornd(1, v);
    h : = W * v +  bias;
    reconstruction := transpose(W) * h + vbias;
    where v is ground truth.
    This function calculates MSE between ground truth and reconstrucion.
    """

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "EVALUATOR")
        kwargs["error_function_averaged"] = kwargs.get(
            "error_function_averaged", True)
        super(EvaluatorRBM, self).__init__(workflow, **kwargs)
        self.error_function_averaged = kwargs.get("error_function_averaged",
                                                  True)
        self.output = None  # formats.Vector()
        self.err_output = 0
        self.batch_size = 0
        self.ground_truth = None
        self.krn_constants_i_ = None
        self.krn_constants_f_ = None
        self.first_minibatch = True
        self.result = 0
        self.demand("output", "batch_size")

    def rbm_energy(self):
        v = matlab_binornd(1, self.ground_truth.mem)
        v = v[0:self.batch_size, :]
        h = numpy.dot(v, self.weights.mem.transpose())
        h += self.bias.mem
        h_plus = h[:]
        h_plus[h < 0] = 0
        Es1 = numpy.dot(v, self.vbias.mem.transpose())
        Es2 = numpy.log(numpy.exp(-h_plus) + numpy.exp(h - h_plus)) + h_plus
        Es2 = numpy.sum(Es2, 1)
        Es2 = Es2.reshape(Es2.shape[0], 1)
        Es = -(Es1 + Es2)
        Es = Es[: self.batch_size, 0]
        Emin = numpy.min(Es)
        Emax = numpy.max(Es)
        E = numpy.mean(Es)
        Es = numpy.sum(Es)
        return E, Emin, Emax, Es

    def initialize(self, device, **kwargs):
        super(EvaluatorRBM, self).initialize(device, **kwargs)
        n_minibatches = math.ceil(self.class_lengths[2] /
                                  (1.0 * self.max_minibatch_size))
        self.reconstruction_error = numpy.zeros((1, n_minibatches *
                                                 self.max_epochs),
                                                dtype=numpy.float64)

        self.reconstruction_iter = 0

    def ocl_run(self):
        pass

    def cpu_run(self):
        self.ground_truth.map_read()
        ground_truth = self.ground_truth.mem.copy()
        for v in (self.input, self.weights, self.vbias):
            v.map_read()
        # input is h0 numpy array of float numbers
        mem = matlab_binornd(1, self.input.mem[:self.batch_size, :])
        vr = numpy.dot(mem, self.weights.mem)
        vr += self.vbias.mem
        vr = 1 / (1 + numpy.exp(-vr))
        ground_truth = ground_truth[0:self.batch_size, :]
        rerr_cur = numpy.sum(numpy.sum((ground_truth - vr) ** 2, 1)) / \
            self.batch_size

        if self.minibatch_class == 2:
            self.reconstruction_error[0, self.reconstruction_iter] = rerr_cur
            self.reconstruction_iter += 1
