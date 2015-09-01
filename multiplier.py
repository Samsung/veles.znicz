# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 3, 2015

Multiplies two arrays pointwise.

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
from zope.interface import implementer

from veles.accelerated_units import (
    IOpenCLUnit, ICUDAUnit, INumpyUnit, AcceleratedUnit)
from veles.memory import Array


@implementer(ICUDAUnit, IOpenCLUnit, INumpyUnit)
class Multiplier(AcceleratedUnit):
    """Multiplies two vectors pointwise.
    """
    def __init__(self, workflow, **kwargs):
        super(Multiplier, self).__init__(workflow, **kwargs)
        self.output = Array()
        self.demand("x", "y")

    def initialize(self, device, **kwargs):
        if ((not self.output and (self.x or self.y)) or
                (self.x and self.output.shape[0] != self.x.shape[0]) or
                (self.y and self.output.shape[0] != self.y.shape[0])):
            self.output.reset(
                numpy.zeros_like(self.x.mem if self.x else self.y.mem))
        if not self.x or not self.y:
            return True
        super(Multiplier, self).initialize(device, **kwargs)
        assert self.output.shape == self.x.shape == self.y.shape
        self.init_vectors(self.x, self.y, self.output)

    def init_unpickled(self):
        super(Multiplier, self).init_unpickled()
        self.sources_["multiplier"] = {}

    def _gpu_init(self):
        self.build_program({"OUTPUT_SIZE": self.output.size},
                           "%s_%d" %
                           (self.__class__.__name__, self.output.size),
                           dtype=self.x.dtype)
        self.assign_kernel("multiply_forward")
        self.set_args(self.x, self.y, self.output)

    def cuda_init(self):
        self._gpu_init()
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (
            int(numpy.ceil(self.output.size / block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def ocl_init(self):
        self._gpu_init()
        self._global_size = (self.output.size, 1, 1)
        self._local_size = None

    def numpy_init(self):
        pass  # nothing to init

    def _gpu_run(self):
        self.unmap_vectors(self.x, self.y, self.output)
        self.execute_kernel(self._global_size, self._local_size)

    def cuda_run(self):
        self._gpu_run()

    def ocl_run(self):
        self._gpu_run()

    def numpy_run(self):
        self.x.map_read()
        self.y.map_read()
        self.output.map_invalidate()
        numpy.multiply(self.x.mem, self.y.mem, self.output.mem)


@implementer(ICUDAUnit, IOpenCLUnit, INumpyUnit)
class GDMultiplier(AcceleratedUnit):
    """Gradient descent for Multiplier.
    """
    def __init__(self, workflow, **kwargs):
        super(GDMultiplier, self).__init__(workflow, **kwargs)
        self.err_x = Array()
        self.err_y = Array()
        self.demand("x", "y", "err_output")

    def initialize(self, device, **kwargs):
        super(GDMultiplier, self).initialize(device, **kwargs)

        if self.err_x:
            assert self.err_x.shape[1:] == self.x.shape[1:]
        if not self.err_x or self.err_x.shape[0] != self.x.shape[0]:
            self.err_x.reset(numpy.zeros_like(self.x.mem))

        if self.err_y:
            assert self.err_y.shape[1:] == self.y.shape[1:]
        if not self.err_y or self.err_y.shape[0] != self.y.shape[0]:
            self.err_y.reset(numpy.zeros_like(self.y.mem))

        self.init_vectors(self.err_x, self.err_y,
                          self.x, self.y, self.err_output)

    def init_unpickled(self):
        super(GDMultiplier, self).init_unpickled()
        self.sources_["multiplier"] = {}

    def _gpu_init(self):
        self.build_program({"OUTPUT_SIZE": self.err_output.size},
                           "%s_%d" %
                           (self.__class__.__name__, self.err_output.size),
                           dtype=self.x.dtype)
        self.assign_kernel("multiply_backward")
        self.set_args(self.x, self.y, self.err_output, self.err_x, self.err_y)

    def cuda_init(self):
        self._gpu_init()
        block_size = self.device.suggest_block_size(self._kernel_)
        self._global_size = (
            int(numpy.ceil(self.err_output.size / block_size)), 1, 1)
        self._local_size = (block_size, 1, 1)

    def ocl_init(self):
        self._gpu_init()
        self._global_size = (self.err_output.size, 1, 1)
        self._local_size = None

    def numpy_init(self):
        pass  # nothing to init

    def _gpu_run(self):
        self.unmap_vectors(self.x, self.y, self.err_output,
                           self.err_x, self.err_y)
        self.execute_kernel(self._global_size, self._local_size)

    def cuda_run(self):
        self._gpu_run()

    def ocl_run(self):
        self._gpu_run()

    def numpy_run(self):
        self.x.map_read()
        self.y.map_read()
        self.err_output.map_read()
        self.err_x.map_invalidate()
        self.err_y.map_invalidate()
        numpy.multiply(self.err_output.mem, self.y.mem, self.err_x.mem)
        numpy.multiply(self.err_output.mem, self.x.mem, self.err_y.mem)
