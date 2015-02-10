"""
Created on Jul 1, 2014

Deconvolutional layer.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import numpy
from zope.interface import implementer

from veles.config import root
from veles.accelerated_units import IOpenCLUnit
from veles.memory import roundup, Vector
import veles.znicz.nn_units as nn_units
import veles.error as error
from veles.distributable import TriviallyDistributable


@implementer(IOpenCLUnit)
class Deconv(TriviallyDistributable, nn_units.Forward):
    # TriviallyDistributable overrides nn_units.Forward IDistributable
    """Deconvolutional layer for simple convolutional layer
    with linear activation and without bias.

    Must be assigned before initialize():
        input
        weights
        get_output_shape_from

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        input: input as batch of multichannel interleaved images.
        output: output as batch of multichannel interleaved images.
        weights: matrix of weights.
        get_output_shape_from: Vector to get output shape from.
        n_kernels: number of convolutional kernels
                   in the corresponding convolutional layer.
        kx: kernel width.
        ky: kernel height.
        sliding: tuple of kernel sliding (by x-axis, by y-axis),
                 kx, ky MUST be a multiple of sliding to avoid irregularities.
        padding: tuple of virtual sample padding (left, top, right, bottom),
                 will be computed automatically based on sliding.
        weights_transposed: assume weights matrix as a transposed one.
        unsafe_padding: flag to enable unsafe padding and/or sliding.
    """

    MAPPING = {"deconv"}

    @staticmethod
    def compute_padding(sx, sy, kx, ky, sliding):
        """Computes required padding.
        """
        if sliding[0] > (ky >> 1) or sliding[1] > (kx >> 1):
            raise error.BadFormatError(
                "sliding should not be greater than half of the kernel size")
        if ky % sliding[0] != 0 or kx % sliding[1] != 0:
            raise error.BadFormatError(
                "Kernel size should be multiple of sliding")
        return (kx - sliding[1], ky - sliding[0],
                kx - sx % sliding[1] if sx % sliding[1] != 0
                else kx - sliding[1],
                ky - sy % sliding[0] if sy % sliding[0] != 0
                else ky - sliding[0])

    def __init__(self, workflow, **kwargs):
        try:
            n_kernels = kwargs["n_kernels"]
            kx = kwargs["kx"]
            ky = kwargs["ky"]
        except KeyError:
            raise KeyError("n_kernels, kx and ky are required parameters")
        super(Deconv, self).__init__(workflow, **kwargs)
        self.n_kernels = n_kernels
        self.kx = kx
        self.ky = ky
        self.padding = kwargs.get("padding")
        self.sliding = tuple(kwargs.get("sliding", (1, 1)))
        self.bias = None
        self.get_output_shape_from = None
        self.exports.extend(("kx", "ky", "n_kernels", "padding", "sliding"))
        self.demand("input", "weights", "get_output_shape_from")
        self.unsafe_padding = kwargs.get("unsafe_padding", False)
        self.hits = Vector()
        self.krn_clear_output_ = None
        self._global_size = None
        self._local_size = None

    def init_unpickled(self):
        super(Deconv, self).init_unpickled()
        self.sources_["deconv/forward"] = {}

    def initialize(self, device, **kwargs):
        super(Deconv, self).initialize(device, **kwargs)

        if self.bias is not None:
            raise error.BadFormatError("bias should not be set")
        if self.input.mem is None:
            raise error.BadFormatError("input should be assigned")
        if (len(self.input.shape) != 4 or
                self.input.shape[3] != self.n_kernels):
            raise error.BadFormatError(
                "Incorrectly shaped input encountered")
        if self.weights.mem is None:
            raise error.BadFormatError("weights should be assigned")
        weights_shape = (list(
            self.weights.shape[i] for i in range(
                len(self.weights.shape) - 1, -1, -1))
            if self.weights_transposed else list(self.weights.shape))
        if (len(weights_shape) != 2 or
                weights_shape[0] != self.n_kernels or
                weights_shape[1] % (self.kx * self.ky) != 0):
            raise error.BadFormatError(
                "Incorrectly shaped weights encountered")

        dtype = self.input.mem.dtype

        output_shape = list(self.get_output_shape_from.shape)
        if len(output_shape) != 4:
            raise error.BadFormatError("Incorrect get_output_shape_from shape")
        if output_shape[0] != self.input.shape[0]:
            raise error.BadFormatError(
                "get_output_shape_from.shape[0] != input.shape[0]")

        try:
            padding = Deconv.compute_padding(
                output_shape[2], output_shape[1],
                self.kx, self.ky, self.sliding)
            if self.padding is None:
                self.padding = padding
            elif self.padding != padding:
                raise error.BadFormatError("Expected padding %s got %s" %
                                           (str(padding), str(self.padding)))
        except error.BadFormatError:
            if not self.unsafe_padding:
                raise
            self.warning("Using unsafe padding of %s", str(self.padding))
            if (not self.hits or
                    self.hits.size != int(numpy.prod(output_shape))):
                self.hits.reset(numpy.zeros(output_shape, dtype=numpy.int32))

        if not self.output:
            self.output.reset(numpy.zeros(output_shape, dtype=dtype))
        else:
            assert self.output.shape == output_shape

        self.init_vectors(self.input, self.weights, self.output, self.hits)

    def ocl_init(self):
        dtype = self.input.mem.dtype
        output_shape = list(self.get_output_shape_from.shape)
        weights_shape = (list(
            self.weights.shape[i] for i in range(
                len(self.weights.shape) - 1, -1, -1))
            if self.weights_transposed else list(self.weights.shape))

        sx = output_shape[2]
        sy = output_shape[1]
        n_channels = output_shape[3]

        kernel_applies_count = self.input.size // self.n_kernels
        a_width = kernel_applies_count
        b_width = weights_shape[1]
        block_size = self.device.device_info.get_block_size(
            kernel="deconv", dtype=self.input.dtype)

        defines = {
            "USE_ATOMICS": 1,
            "WEIGHTS_TRANSPOSED": int(self.weights_transposed),
            "BATCH": output_shape[0],
            "SX": sx,
            "SY": sy,
            "N_CHANNELS": n_channels,
            "KX": self.kx,
            "KY": self.ky,
            "N_KERNELS": self.n_kernels,
            "PAD_LEFT": self.padding[0],
            "PAD_TOP": self.padding[1],
            "PAD_RIGHT": self.padding[2],
            "PAD_BOTTOM": self.padding[3],
            "SLIDE_X": self.sliding[0],
            "SLIDE_Y": self.sliding[1],
            "USE_HITS": int(bool(self.hits)),
            "BLOCK_SIZE": block_size
        }

        self.build_program(
            defines, "%s/%s_%d_%dx%dx%d_%dx%d_%d" % (
                root.common.cache_dir, self.__class__.__name__,
                self.input.shape[0],
                output_shape[2], output_shape[1], output_shape[3],
                self.kx, self.ky, self.n_kernels), dtype=dtype)

        self.assign_kernel("feed_layer")
        self.set_args(self.input, self.weights, self.output)

        self.krn_clear_output_ = self.get_kernel("clear_output")
        self.krn_clear_output_.set_arg(0, self.output.devmem)

        self._global_size = [
            roundup(b_width, block_size),
            roundup(a_width, block_size)]
        self._local_size = [block_size, block_size]

        if self.hits:
            self.krn_apply_hits_ = self.get_kernel("apply_hits")
            self.krn_apply_hits_.set_args(self.output.devmem, self.hits.devmem)

            self.krn_clear_hits_ = self.get_kernel("clear_hits")
            self.krn_clear_hits_.set_arg(0, self.hits.devmem)

            self.set_arg(3, self.hits)

    def ocl_run(self):
        self.output.unmap()
        self.input.unmap()
        self.weights.unmap()
        self.execute_kernel([self.output.size], None, self.krn_clear_output_)
        if self.hits:
            self.hits.unmap()
            self.execute_kernel([self.hits.size], None, self.krn_clear_hits_)
            self.execute_kernel(self._global_size, self._local_size)
            self.execute_kernel([self.hits.size], None, self.krn_apply_hits_)
        else:
            self.execute_kernel(self._global_size, self._local_size)

    def cpu_run(self):
        raise RuntimeError("Not implemented")
