# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 27, 2013

Simple convolutional layer (:class:`Conv`) and conv layer with subsequent \
    activations (:class:`ConvRELU`, :class:`ConvStrictRELU`, :class:`ConvTanh`)

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

import cuda4py.blas as cublas
import math
from math import pi
import numpy
import time
from zope.interface import implementer

from veles.accelerated_units import IOpenCLUnit, ICUDAUnit, INumpyUnit
from veles.compat import from_none
import veles.error as error
from veles.memory import reshape_transposed
from veles.units import Unit
import veles.ocl_blas as ocl_blas
import veles.znicz.nn_units as nn_units


class ConvolutionalBase(Unit):
    hide_from_registry = True
    CONV_ATTRS = ("n_kernels", "kx", "ky", "sliding", "padding", "unpack_size")

    def __init__(self, workflow, **kwargs):
        super(ConvolutionalBase, self).__init__(workflow, **kwargs)
        self.demand(*self.CONV_ATTRS)

    def link_conv_attrs(self, other):
        self.link_attrs(other, *self.CONV_ATTRS)
        return self


@implementer(IOpenCLUnit, ICUDAUnit, INumpyUnit)
class Conv(ConvolutionalBase, nn_units.NNLayerBase):
    """Convolutional forward propagation with linear activation f(x) = x.

    Must be assigned before initialize():
        input

    Updates after run():
        output

    Creates within initialize():
        weights
        bias
        output

    Attributes:
        input: input as batch of multichannel interleaved images.
        output: output as batch of multichannel interleaved images.
        weights: matrix of weights.
        bias: bias.
        n_kernels: number of convolutional kernels.
        kx: kernel width.
        ky: kernel height.
        padding: tuple of virtual sample padding (left, top, right, bottom).
        sliding: tuple of kernel sliding (by x-axis, by y-axis).
        weights_filling: what weight filling to use: `"uniform"` for uniform
            random distribution, `"normal"` for normal distribution,
            `"gabor"` for using Gabor kernels.
        weights_stddev: standard deviation of normal or Gabor weight fillings

        rand: rnd.Rand() object for initial weights generation.
        activation_mode: activation define for OpenCL source.
        weights_transposed: assume weights matrix as a transposed one.
                            NOTE: only access order will be affected,
                            not a shape.
    """

    MAPPING = {"conv"}

    def __init__(self, workflow, **kwargs):
        super(Conv, self).__init__(workflow, **kwargs)
        try:
            self.n_kernels = kwargs["n_kernels"]
            self.kx = kwargs["kx"]
            self.ky = kwargs["ky"]
        except KeyError:
            raise from_none(KeyError(
                "n_kernels, kx and ky are required parameters"))
        self.padding = tuple(
            kwargs.get("padding", (0, 0, 0, 0)))  # Left Top Right Bottom
        self.sliding = tuple(kwargs.get("sliding", (1, 1)))  # X Y
        self.activation_mode = "ACTIVATION_LINEAR"
        self.exports.extend(("activation_mode", "kx", "ky", "n_kernels",
                             "padding", "sliding"))
        self._global_size = None
        self._local_size = None

        # Image count to unpack at once
        self.unpack_size = kwargs.get("unpack_size", 16)

    def init_unpickled(self):
        super(Conv, self).init_unpickled()
        self.sources_["conv/forward"] = {}

    def get_weights_magnitude(self):
        """
        Returns: weights magnitude for initial random distribution,
                 such that activation function will be near maximum
                 if all input values are at their supposed max value.
        """
        n_channels = (self.input.size // (self.input.shape[0] *
                      self.input.shape[1] * self.input.shape[2]))
        vle = (1.0 / self.input.max_supposed /
               numpy.sqrt(self.kx * self.ky * n_channels))
        if self.weights_filling == "gaussian":
            vle /= 3
        return vle

    def initialize(self, device, **kwargs):
        super(Conv, self).initialize(device, **kwargs)

        if self.weights_stddev is None:
            self.weights_stddev = min(self.get_weights_magnitude(), 0.05)
        if self.bias_stddev is None:
            self.bias_stddev = self.weights_stddev

        self._batch_size = self.input.shape[0]
        self._sy = self.input.shape[1]
        self._sx = self.input.shape[2]
        self._n_channels = (self.input.size //
                            (self._batch_size * self._sx * self._sy))
        self._kx_app = (
            1 + ((self._sx - self.kx +
                  self.padding[0] + self.padding[2]) // self.sliding[0]))
        self._ky_app = (
            1 + ((self._sy - self.ky +
                  self.padding[1] + self.padding[3]) // self.sliding[1]))
        self._kernel_app_per_image = self._kx_app * self._ky_app
        self._kernel_size = self.kx * self.ky * self._n_channels

        self._fill_weights()
        self._fill_biases()

        output_shape = (self._batch_size, self._ky_app, self._kx_app,
                        self.n_kernels)
        if self.output:
            assert self.output.shape[1:] == output_shape[1:]
        if not self.output or output_shape[0] != self.output.shape[0]:
            self.output.reset(numpy.zeros(output_shape, self.input.dtype))

        assert self._kernel_app_per_image * self.n_kernels == \
            self.output.sample_size

        self.init_vectors(self.input, self.output, self.weights, self.bias)

    def _gpu_init(self, blas_class):
        dtype = self.input.dtype

        defines = {
            self.activation_mode: 1,
            "WEIGHTS_TRANSPOSED": int(self.weights_transposed),
            "INCLUDE_BIAS": int(self.include_bias),
            "BATCH": self._batch_size,
            "SX": self._sx,
            "SY": self._sy,
            "N_CHANNELS": self._n_channels,
            "KX": self.kx,
            "KY": self.ky,
            "N_KERNELS": self.n_kernels,
            "PAD_LEFT": self.padding[0],
            "PAD_TOP": self.padding[1],
            "PAD_RIGHT": self.padding[2],
            "PAD_BOTTOM": self.padding[3],
            "SLIDE_X": self.sliding[0],
            "SLIDE_Y": self.sliding[1],
            "OUTPUT_SIZE": self.output.size,
            "BIAS_SIZE": self.n_kernels
        }

        self.build_program(
            defines, "%s_%d_%dx%dx%d_%dx%d_%d" % (
                self.__class__.__name__, self._batch_size,
                self._sx, self._sy, self._n_channels,
                self.kx, self.ky, self.n_kernels), dtype=dtype)

        self.gemm_ = blas_class.gemm(dtype)
        self.np_one = numpy.ones(1, dtype=dtype)
        self.np_zero = numpy.zeros(1, dtype=dtype)
        self._const_i = numpy.zeros(1, dtype=numpy.int64)

        self.assign_kernel("Unpack1D")
        unpack_bytes = (self._kernel_app_per_image * self.unpack_size *
                        self._kernel_size * self.input.itemsize)
        self.device.request_temp_buffer(unpack_bytes)

        if self.include_bias or self.activation_mode != "ACTIVATION_LINEAR":
            self._krn_bias_ = self.get_kernel("apply_bias_with_activation")
            self._krn_bias_.set_args(self.output.devmem, self.bias.devmem)
        else:
            self._krn_bias_ = None

    def ocl_init(self):
        ocl_blas.OCLBLAS.attach_to_device(self.device)
        self._gpu_init(ocl_blas.OCLBLAS)

        self._global_size_unpack = lambda size: (size,)
        self._local_size_unpack = None

        if self._krn_bias_ is not None:
            self._global_size_bias = (self.output.size,)
            self._local_size_bias = None

        self._process_subblock = self._ocl_process_subblock

        self.set_arg(0, self.input)

    def cuda_init(self):
        self._gpu_init(cublas.CUBLAS)

        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size_unpack = (
            lambda size: (int(numpy.ceil(size / block_size)), 1, 1))
        self._local_size_unpack = (block_size, 1, 1)

        if self._krn_bias_ is not None:
            block_size = self.device.suggest_block_size(self._krn_bias_)
            self._global_size_bias = (
                int(numpy.ceil(self.output.size / block_size)), 1, 1)
            self._local_size_bias = (block_size, 1, 1)

        self._process_subblock = self._cuda_process_subblock

    def ocl_run(self):
        self.gpu_run()

    def cuda_run(self):
        self.gpu_run()

    def gpu_run(self):
        self.unmap_vectors(self.input, self.weights, self.bias, self.output)
        unpack_data = self.device.get_temp_buffer()
        for i in range(0, self._batch_size, self.unpack_size):
            self._process_subblock(
                i, min(self._batch_size - i, self.unpack_size), unpack_data)
        if self.include_bias or self.activation_mode != "ACTIVATION_LINEAR":
            self.execute_kernel(self._global_size_bias, self._local_size_bias,
                                self._krn_bias_)

    def _cuda_process_subblock(self, start_image, image_count, unpack_data):
        self._kernel_.set_arg(
            0, int(self.input.devmem) +
            start_image * self.input.sample_size * self.input.itemsize)
        self._kernel_.set_arg(1, unpack_data)
        unpack_side = self._kernel_app_per_image * image_count
        limit = unpack_side * self._kernel_size
        if self._const_i is not None:
            self._const_i[0] = limit
            self._kernel_.set_arg(2, self._const_i)
        self.execute_kernel(self._global_size_unpack(limit),
                            self._local_size_unpack)
        output_offs = (start_image * self.output.sample_size *
                       self.output.itemsize)
        self.gemm_(
            self.device.blas, cublas.CUBLAS_OP_N if self.weights_transposed
            else cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
            self.weights_shape[0], unpack_side, self._kernel_size,
            self.np_one, self.weights.devmem, unpack_data,
            self.np_zero, int(self.output.devmem) + output_offs)

    def _ocl_process_subblock(self, start_image, image_count, unpack_data):
        self._const_i[0] = start_image * self.input.sample_size
        self._kernel_.set_arg(1, unpack_data)
        self._kernel_.set_arg(2, self._const_i)
        unpack_side = self._kernel_app_per_image * image_count
        limit = unpack_side * self._kernel_size
        self.execute_kernel(self._global_size_unpack(limit),
                            self._local_size_unpack)
        output_offs = start_image * self.output.sample_size
        self.gemm_(
            self.device.blas, cublas.CUBLAS_OP_N if self.weights_transposed
            else cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
            self.weights_shape[0], unpack_side, self._kernel_size,
            self.np_one, self.weights.devmem, unpack_data,
            self.np_zero, self.output.devmem, offsetC=output_offs)

    def numpy_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.input.map_read()
        self.weights.map_read()
        self.bias.map_read()
        self.output.map_invalidate()

        sx_full = self.padding[0] + self._sx + self.padding[2]
        sy_full = self.padding[1] + self._sy + self.padding[3]
        nx = (sx_full - self.kx) // self.sliding[0] + 1
        ny = (sy_full - self.ky) // self.sliding[1] + 1

        weights = (reshape_transposed(self.weights.mem)
                   if self.weights_transposed else self.weights.mem)

        assert self.kx >= 0 and self.ky >= 0
        for batch, _ in ((batch, ch)
                         for batch in range(self._batch_size)
                         for ch in range(self._n_channels)):
            for k, kernel in enumerate(weights):
                for i, j in ((i, j) for i in range(ny) for j in range(nx)):
                    full_i1 = i * self.sliding[1]
                    full_i2 = full_i1 + self.ky
                    full_j1 = j * self.sliding[0]
                    full_j2 = full_j1 + self.kx
                    in_i1 = min(max(full_i1 - self.padding[1], 0), self._sy)
                    in_i2 = min(max(full_i2 - self.padding[1], 0), self._sy)
                    in_j1 = min(max(full_j1 - self.padding[0], 0), self._sx)
                    in_j2 = min(max(full_j2 - self.padding[0], 0), self._sx)
                    cut_i1, cut_i2 = (in_i1 - full_i1 + self.padding[1],
                                      in_i2 - full_i1 + self.padding[1])
                    cut_j1, cut_j2 = (in_j1 - full_j1 + self.padding[0],
                                      in_j2 - full_j1 + self.padding[0])
                    if in_i2 - in_i1 > 0 or in_j2 - in_j1 > 0:
                        cut = self.input.mem[batch, in_i1:in_i2, in_j1:in_j2]
                        kernel_3d = kernel.reshape(self.ky, self.kx,
                                                   self._n_channels)
                        cutted_kernel = kernel_3d[cut_i1:cut_i2,
                                                  cut_j1:cut_j2, :]
                        assert cut.size == cutted_kernel.size
                        conv = numpy.sum(numpy.multiply(cut.ravel(),
                                                        cutted_kernel.ravel()))
                        self.output.mem[batch, i, j, k] = conv
        # add bias and apply activation function
        self.apply_activation()

    def run(self):
        t1 = time.time()
        retval = super(Conv, self).run()
        if retval:
            return retval
        self.print_debug_data(t1)

    def apply_activation(self):
        """Add bias and apply linear activation function.
        """
        assert self.activation_mode == "ACTIVATION_LINEAR"
        if self.include_bias:
            self.output.mem += self.bias.mem

    def _fill_array(self, filling_type, mem, stddev):
        if filling_type == "uniform":
            self.rand.fill(mem, -stddev, stddev)
        elif filling_type == "gaussian":
            self.rand.fill_normal_real(mem, 0, stddev)
        elif filling_type == "constant":
            mem[:] = stddev
        elif filling_type == "gabor":
            self._fill_with_gabor_filters(
                self.n_kernels, (self.ky, self.kx), stddev)
        else:
            raise error.BadFormatError(
                "Invalid filling type: %s" % filling_type)

    def _fill_weights(self):
        """
        Fills initial filter weights according to `weights_filling` attribute.
        Called within ``initialize`` method.
        """
        self.weights_shape = (self.n_kernels,
                              self.kx * self.ky * self._n_channels)
        weights_shape_t = tuple(reversed(self.weights_shape))
        if not self.weights:
            self.weights.reset(numpy.zeros(self.weights_shape,
                                           dtype=self.input.dtype))
            self._fill_array(self.weights_filling, self.weights.mem,
                             self.weights_stddev)
            if self.weights_transposed:
                a = self.weights.mem.transpose().copy()
                self.weights.plain[:] = a.ravel()[:]
                self.weights.shape = weights_shape_t
        else:
            assert (self.weights.shape == weights_shape_t
                    if self.weights_transposed else self.weights_shape)

    def _fill_biases(self):
        """
        Fills filter biases according to `bias_filling` attribute.
        Called within ``initialize`` method.
        """
        if not self.include_bias:
            return
        if not self.bias:
            self.bias.reset(numpy.zeros(self.n_kernels, self.input.dtype))
            self._fill_array(self.bias_filling, self.bias.mem,
                             self.bias_stddev)
        else:
            assert self.bias.size == self.n_kernels

    def _fill_with_gabor_filters(self, n_filters, shape, stddev):
        """
        Fills weights and biases with Gabor filters. Only 96 filters
            are implemented now, others are filled with white noise.

        Args:
            n_filters(int): number of filters
            shape(tuple): shape of each filter
            stddev(float): standard deviation of filtering kernels
        """
        import cv2

        def normalize_image(a):
            a -= a.min()
            mx = a.max()
            if mx:
                a *= 255.0 / mx

        # Gabor  filters
        orientations = [0, pi / 4, pi / 2, 3 * pi / 4]  # tilt of filters
        phase_shifts = [0, pi]  # pi phase shift inverts signal

        size = min(shape)

        kernels_count = 0
        n_chans = self.weights.mem.size // (self.kx * self.ky * self.n_kernels)
        for wavelen_ratio in range(4):  # how much waves should lay in kernel
            for dev_ratio in range(1, 2 * wavelen_ratio + 1):
                for ori in orientations:
                    for phase in phase_shifts:
                        kernel_chan = cv2.getGaborKernel(
                            ksize=shape, sigma=size / dev_ratio / 2,
                            theta=ori, lambd=size / wavelen_ratio,
                            gamma=1, psi=phase)

                        kernel_chan = normalize_image(kernel_chan) * stddev
                        kernel = numpy.zeros(shape=[n_chans, self.kx, self.ky],
                                             dtype=numpy.float64)
                        for chan in range(n_chans):
                            kernel[chan, :] = kernel_chan
                        kernel = kernel.swapaxes(0, 2)
                        self.weights.mem[
                            kernels_count * kernel.size:
                            (kernels_count + 1) * kernel.size] = kernel.ravel()

                        kernels_count += 1
                        if kernels_count == n_filters:
                            return

        # White noise (if more, than 96 filters are required)
        self.rand.fill_normal_real(self.weights.mem[kernels_count:], 0, stddev)


class ConvTanh(Conv):
    """Conv with scaled tanh() activation \
        :math:`f(x) = 1.7159 \\tanh(0.6666 x)`.
    """

    MAPPING = {"conv_tanh"}

    def initialize(self, device, **kwargs):
        self.activation_mode = "ACTIVATION_TANH"
        super(ConvTanh, self).initialize(device=device, **kwargs)
        self.output.max_supposed = 1.7159

    def apply_activation(self):
        """Add bias and apply tanh activation function.
        """
        assert self.activation_mode == "ACTIVATION_TANH"
        for k in range(self.n_kernels):
            for x in numpy.nditer(self.output.mem[:, :, :, k],
                                  op_flags=['readwrite']):
                x[...] = math.tanh((x + self.bias.mem[k]) * 0.6666) * 1.7159


class ConvSigmoid(Conv):
    """Conv with Sigmoid activation \
        :math:`f(x) = 1.0 / (1.0 + exp(x))`.
    """

    MAPPING = {"conv_sigmoid"}

    def initialize(self, device, **kwargs):
        self.activation_mode = "ACTIVATION_SIGMOID"
        super(ConvSigmoid, self).initialize(device=device, **kwargs)
        self.output.max_supposed = 1.0

    def apply_activation(self):
        """Add bias and apply sigmoid activation function.
        """
        assert self.activation_mode == "ACTIVATION_SIGMOID"
        for k in range(self.n_kernels):
            for x in numpy.nditer(self.output.mem[:, :, :, k],
                                  op_flags=['readwrite']):
                x[...] = 1.0 / (1.0 + math.exp(-(x + self.bias.mem[k])))


class ConvRELU(Conv):
    """Conv with smooth RELU activation :math:`f(x) = \\log(1 + \\exp(x))`.
    """

    MAPPING = {"conv_relu"}

    def initialize(self, device, **kwargs):
        self.activation_mode = "ACTIVATION_RELU"
        super(ConvRELU, self).initialize(device=device, **kwargs)
        self.output.max_supposed = 10

    def apply_activation(self):
        """Add bias and apply RELU activation function.
        """
        assert self.activation_mode == "ACTIVATION_RELU"
        for k in range(self.n_kernels):
            for x in numpy.nditer(self.output.mem[:, :, :, k],
                                  op_flags=['readwrite']):
                tmp_val = x + self.bias.mem[k]
                if tmp_val > 15:
                    x[...] = tmp_val
                else:
                    x[...] = math.log(math.exp(tmp_val) + 1)


class ConvStrictRELU(Conv):
    """
    Conv with strict RELU activation :math:`f(x) = \\max(x, 0)`
    (Just like in CAFFE)
    """

    MAPPING = {"conv_str"}

    def initialize(self, device, **kwargs):
        self.activation_mode = "ACTIVATION_STRICT_RELU"
        super(ConvStrictRELU, self).initialize(device=device, **kwargs)
        self.output.max_supposed = 10

    def apply_activation(self):
        """Add bias and apply STRICT_RELU activation function.
        """
        assert self.activation_mode == "ACTIVATION_STRICT_RELU"
        for k in range(self.n_kernels):
            for x in numpy.nditer(self.output.mem[:, :, :, k],
                                  op_flags=['readwrite']):
                tmp_val = x + self.bias.mem[k]
                x[...] = numpy.where(numpy.greater(tmp_val, 0), tmp_val, 0)
