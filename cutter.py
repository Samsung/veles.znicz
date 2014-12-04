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
from veles.units import Unit


class CutterBase(Unit):
    def __init__(self, workflow, padding, *args, **kwargs):
        super(CutterBase, self).__init__(workflow, *args, **kwargs)
        self.padding = padding
        self.demand("input")

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        if value is None:
            raise ValueError("padding may not be None")
        if type(value) not in (tuple, list):
            raise TypeError("padding must either of type tuple or list")
        if len(value) != 4:
            raise ValueError(
                "padding must be of length 4: (left, top, right, bottom)")
        self._padding = value

    def create_stuff(self, prefix):
        setattr(self, "_%s_origin" % prefix, (
            self.padding[0] * self.input.shape[3] * self.input.itemsize,
            self.padding[1], 0))
        self._region = (
            self.output_shape[2] * self.output_shape[3] * self.input.itemsize,
            self.output_shape[1], self.input.shape[0])
        setattr(self, "_%s_row_pitch" % prefix, (
            self.input.shape[2] * self.input.shape[3] * self.input.itemsize))
        setattr(self, "_%s_slice_pitch" % prefix,
                self.input.sample_size * self.input.itemsize)


@implementer(IOpenCLUnit)
class Cutter(nn_units.Forward, CutterBase):
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
        super(Cutter, self).__init__(workflow, **kwargs)
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

        self.input.initialize(self)
        self.output.initialize(self)

        self.create_stuff("src")

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
class GDCutter(nn_units.GradientDescentBase, CutterBase):
    """Gradient descent for Cutter.
    """

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

        self.err_output.initialize(self)
        self.err_input.initialize(self)

        self.create_stuff("dst")

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
