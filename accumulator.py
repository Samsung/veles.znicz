"""
Created on April 9, 2014

Accumulator units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import sys

import veles.error as error
import veles.formats as formats
from veles.mutable import Bool
import veles.units as units


class FixAccumulator(units.Unit):
    """
    Range accumulator.
    """
    def __init__(self, workflow, **kwargs):
        super(FixAccumulator, self).__init__(workflow)
        self.bars = kwargs.get("bars", 200)
        self.type = kwargs.get("type", "relu")
        self.input = None
        self.output = formats.Vector()
        self.reset_flag = Bool(True)
        self.n_bars = [0]
        self.max = 100
        self.min = 0

    def initialize(self, **kwargs):
        super(FixAccumulator, self).initialize(**kwargs)
        self.output.mem = numpy.zeros([self.bars + 2], dtype=numpy.int64)

    def run(self):
        if self.type == "relu":
            self.max = 10000
            self.min = 0
        elif self.type == "tanh":
            self.max = 1.7159
            self.min = -1.7159
        else:
            raise error.ErrBadFormat("Unsupported type %s" % self.type)

        d = self.max - self.min
        if not d:
            return
        self.output.map_write()
        self.input.map_read()
        d = (self.bars - 1) / d
        if self.reset_flag:
            self.output.mem[:] = 0
        self.n_bars[0] = self.bars + 2
        for y in self.input.mem.ravel():
            if y < self.min:
                self.output[0] += 1
                continue
            if y <= self.max and y > self.min:
                i = int(numpy.floor((y - self.min) * d))
                self.output[i] += 1
                continue
            self.output[self.bars + 1] += 1


class RangeAccumulator(units.Unit):
    """Range accumulator.
    """
    def __init__(self, workflow, **kwargs):
        super(RangeAccumulator, self).__init__(workflow)
        self.bars = kwargs.get("bars", 20)
        self.x = []
        self.input = None
        self.first_minibatch = True
        self.y = []
        self.d = 0
        self.reset_flag = Bool(False)
        self.gl_min = sys.float_info.max
        self.gl_max = sys.float_info.min

    def initialize(self, **kwargs):
        super(RangeAccumulator, self).initialize(**kwargs)

    def run(self):
        if self.reset_flag:
            self.x.clear()
            self.y.clear()
            self.gl_max = sys.float_info.min
            self.gl_min = sys.float_info.max
            self.first_minibatch = True
        if self.first_minibatch:
            self.input.map_read()
            in_max = self.input.mem.max()
            in_min = self.input.mem.min()
            self.gl_min = in_min
            self.gl_max = in_max
            d = in_max - in_min
            if not d:
                return
            self.d = d / (self.bars - 1)
            for i in range(0, self.bars):
                self.y.append(0)
                self.x.append(in_min + self.d / 2 + i * self.d)
            for inp in self.input.mem.ravel():
                i = int(numpy.floor((inp - in_min) / self.d))
                self.y[i] += 1
            self.first_minibatch = False
        else:
            self.input.map_read()
            in_max = self.input.mem.max()
            in_min = self.input.mem.min()
            x_max = numpy.max(self.x)
            x_min = numpy.min(self.x)
            self.gl_min = min(in_min, self.gl_min)
            self.gl_max = max(in_max, self.gl_max)
            if in_max > x_max:
                diff = in_max - x_max
                i_diff = int(numpy.ceil(diff / self.d))
                for i in range(1, i_diff + 1):
                    self.y.append(0)
                    self.x.append(x_max - self.d / 2 + i * self.d)
            if in_min < x_min:
                diff = x_min - in_min
                i_diff = int(numpy.floor(diff / self.d))
                for i in range(1, i_diff + 1):
                    self.y.insert(0, 0)
                    self.x.insert(0, x_min + self.d / 2 - i * self.d)
            for inp in self.input.mem.ravel():
                i = int(numpy.floor((inp - numpy.min(self.x)) / self.d))
                self.y[i] += 1
        """
        if len(self.x) > self.bars:
            segm = int(len(self.x) / self.bars)
            residue = len(self.x) - segm * self.bars
            sum_x = 0
            x_new = []
            y_new = []
            for i in range(0, len(self.x) - residue, segm):
                for j in range(0, segm):
                    sum_x += self.x(i + j)
                    y += self.y(i + j)
                x = sum_x / segm
                x_new.append(x)
                y_new.append(y)
            for j in range(0, residue):
                sum_x += self.x(len(self.x) - residue + j)
                y += self.y(len(self.x) - residue + j)
            x = sum_x / residue
            x_new.append(x)
            y_new.append(y)
            self.x = self.x_new[:]
            self.y = self.y_new{:]
        """

