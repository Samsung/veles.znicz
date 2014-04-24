"""
Created on April 9, 2014

Accumulator units.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy

import veles.config as config
import veles.formats as formats
from veles.mutable import Bool
import veles.units as units


class RangeAccumulator(units.Unit):
    """Range accumulator.
    """
    def __init__(self, workflow, **kwargs):
        super(RangeAccumulator, self).__init__(workflow)
        bars = kwargs.get("bars", config.get(config.root.n_bars, 30))
        kwargs["bars"] = bars
        self.input = None
        self.output = formats.Vector()
        self.reset_flag = Bool(True)
        self.n_bars = [0]
        self.bars = bars
        self.max = config.get(config.root.accumulator.max, 1.7159)
        self.min = config.get(config.root.accumulator.min, -1.7159)

    def initialize(self):
        super(RangeAccumulator, self).initialize()
        self.output.v = numpy.zeros([self.bars + 2], dtype=numpy.int64)

    def run(self):
        d = self.max - self.min
        if not d:
            return
        self.output.map_write()
        self.input.map_read()
        d = (self.bars - 1) / d
        if self.reset_flag:
            self.output.v[:] = 0
        self.n_bars[0] = self.bars + 2
        for y in self.input.v.ravel():
            if y < self.min:
                self.output[0] += 1
                continue
            if y <= self.max and y > self.min:
                i = int(numpy.floor((y - self.min) * d))
                self.output[i] += 1
                continue
            self.output[self.bars + 1] += 1
