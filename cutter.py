"""
Created on Aug 4, 2014

Cutter unit.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import numpy
from zope.interface import implementer

from veles.opencl_units import IOpenCLUnit

import veles.error as error
import veles.formats as formats
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit)
class Cutter(nn_units.Forward):
    """Cuts rectangular area from an input.

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
        padding = kwargs.get("padding")
        if (padding is None or type(padding) not in (tuple, list) or
                len(padding) != 4):
            raise KeyError(
                "padding is a required parameter: tuple or list:"
                "(left, top, right, bottom)")
        super(Cutter, self).__init__(workflow, **kwargs)
        self.padding = tuple(padding)
        self.exports.append("padding")

    def initialize(self, device, **kwargs):
        if not self.input or len(self.input.shape) != 4:
            raise error.BadFormatError(
                "input should be assigned and have shape of 4: "
                "(n_samples, sy, sx, n_channels)")
        if self.padding[0] < 0 or self.padding[1] < 0:
            raise error.BadFormatError(
                "padding[0], padding[1] should not be less than zero")
        super(Cutter, self).initialize(device=device, **kwargs)

        sh = list(self.input.shape)
        sh[2] -= self.padding[0] + self.padding[2]
        sh[1] -= self.padding[1] + self.padding[3]
        if sh[2] <= 0 or sh[1] <= 0:
            raise error.BadFormatError("Resulted output shape is empty")
        output_size = int(numpy.prod(sh))
        self.output_shape = sh

        if not self.output or self.output.size != output_size:
            self.output.reset()
            self.output.mem = numpy.zeros(sh, dtype=self.input.dtype)

        self.input.initialize(self.device)
        self.output.initialize(self.device)

        self._src_origin = (
            self.padding[0] * self.input.shape[3] * self.input.itemsize,
            self.padding[1], 0)
        self._region = (
            self.output_shape[2] * self.output_shape[3] * self.input.itemsize,
            self.output_shape[1], self.input.shape[0])
        self._src_row_pitch = (
            self.input.shape[2] * self.input.shape[3] * self.input.itemsize)
        self._src_slice_pitch = self.input.sample_size * self.input.itemsize

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        self.output.unmap()
        self.input.unmap()
        self.device.queue_.copy_buffer_rect(
            self.input.devmem, self.output.devmem,
            self._src_origin, (0, 0, 0), self._region,
            self._src_row_pitch, self._src_slice_pitch,
            need_event=False)

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.output.map_invalidate()
        self.input.map_read()
        out = formats.reshape(self.output.mem, self.output_shape)
        inp = self.input.mem
        out[:, :, :, :] = inp[
            :, self.padding[1]:self.padding[1] + self.output_shape[1],
            self.padding[0]:self.padding[0] + self.output_shape[2], :]


@implementer(IOpenCLUnit)
class GDCutter(nn_units.GradientDescentBase):
    """Gradient descent for Cutter.
    """
    def __init__(self, workflow, **kwargs):
        padding = kwargs.get("padding")
        if (padding is None or type(padding) not in (tuple, list) or
                len(padding) != 4):
            raise KeyError(
                "padding is a required parameter: tuple or list:"
                "(left, top, right, bottom)")
        super(GDCutter, self).__init__(workflow, **kwargs)
        self.padding = tuple(padding)

    def initialize(self, device, **kwargs):
        if not self.input or len(self.input.shape) != 4:
            raise error.BadFormatError(
                "input should be assigned and have shape of 4: "
                "(n_samples, sy, sx, n_channels)")
        if self.padding[0] < 0 or self.padding[1] < 0:
            raise error.BadFormatError(
                "padding[0], padding[1] should not be less than zero")
        if not self.err_output:
            raise error.BadFormatError("err_output should be assigned")
        super(GDCutter, self).initialize(device=device, **kwargs)

        sh = list(self.input.shape)
        sh[2] -= self.padding[0] + self.padding[2]
        sh[1] -= self.padding[1] + self.padding[3]
        if sh[2] <= 0 or sh[1] <= 0:
            raise error.BadFormatError("Resulted output shape is empty")
        output_size = int(numpy.prod(sh))
        if self.err_output.size != output_size:
            raise error.BadFormatError(
                "Computed err_output size differs from an assigned one")
        self.output_shape = sh

        if not self.err_input or self.err_input.size != self.input.size:
            self.err_input.reset()
            self.err_input.mem = numpy.zeros_like(self.input.mem)

        self.err_output.initialize(self.device)
        self.err_input.initialize(self.device)

        self._dst_origin = (
            self.padding[0] * self.input.shape[3] * self.input.itemsize,
            self.padding[1], 0)
        self._region = (
            self.output_shape[2] * self.output_shape[3] * self.input.itemsize,
            self.output_shape[1], self.input.shape[0])
        self._dst_row_pitch = (
            self.input.shape[2] * self.input.shape[3] * self.input.itemsize)
        self._dst_slice_pitch = self.input.sample_size * self.input.itemsize

        if self.device is not None:
            GDCutter.ocl_init(self, device)

    def ocl_init(self, device):
        self.cl_sources_["cutter.cl"] = {}
        self.build_program(dtype=self.err_input.dtype)
        self.assign_kernel("clear_err_input")
        self.set_args(self.err_input)

    def ocl_run(self):
        """Forward propagation from batch on GPU.
        """
        self.err_output.unmap()
        self.err_input.unmap()
        self.execute_kernel([self.err_input.size], None)
        self.device.queue_.copy_buffer_rect(
            self.err_output.devmem, self.err_input.devmem,
            (0, 0, 0), self._dst_origin, self._region,
            0, 0, self._dst_row_pitch, self._dst_slice_pitch,
            need_event=False)

    def cpu_run(self):
        """Forward propagation from batch on CPU only.
        """
        self.err_input.map_invalidate()
        self.err_output.map_read()
        out = formats.reshape(self.err_output.mem, self.output_shape)
        inp = self.err_input.mem
        inp[:] = 0
        inp[:, self.padding[1]:self.padding[1] + self.output_shape[1],
            self.padding[0]:self.padding[0] + self.output_shape[2], :] = out
