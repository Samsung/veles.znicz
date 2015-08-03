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

Adds two arrays pointwise.

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
class Summator(AcceleratedUnit):
    """Multiplies two vectors pointwise.
    """
    def __init__(self, workflow, **kwargs):
        super(Summator, self).__init__(workflow, **kwargs)
        self.output = Array()
        self.demand("x", "y")

    def initialize(self, device, **kwargs):
        super(Summator, self).initialize(device, **kwargs)
        if not self.output:
            self.output.reset(numpy.zeros_like(self.x.mem))
        else:
            assert self.output.shape == self.x.shape
        self.init_vectors(self.x, self.y, self.output)

    def init_unpickled(self):
        super(Summator, self).init_unpickled()
        self.sources_["summator"] = {}

    def _gpu_init(self):
        self.build_program({"OUTPUT_SIZE": self.output.size},
                           "%s_%d" %
                           (self.__class__.__name__, self.output.size),
                           dtype=self.x.dtype)
        self.assign_kernel("add_forward")
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
        numpy.add(self.x.mem, self.y.mem, self.output.mem)


@implementer(ICUDAUnit, IOpenCLUnit, INumpyUnit)
class GDSummator(AcceleratedUnit):
    """Gradient descent for Multiplier.
    """
    def __init__(self, workflow, **kwargs):
        super(GDSummator, self).__init__(workflow, **kwargs)
        self.err_x = Array()
        self.err_y = Array()
        self.demand("err_output")

    def initialize(self, device, **kwargs):
        super(GDSummator, self).initialize(device, **kwargs)
        if not self.err_x:
            self.err_x.reset(numpy.zeros_like(self.err_output.mem))
        else:
            assert self.err_x.shape == self.err_output.shape
        if not self.err_y:
            self.err_y.reset(numpy.zeros_like(self.err_output.mem))
        else:
            assert self.err_y.shape == self.err_output.shape
        self.init_vectors(self.err_x, self.err_y, self.err_output)

    def cuda_init(self):
        pass  # nothing to init

    def ocl_init(self):
        pass  # nothing to init

    def numpy_init(self):
        pass  # nothing to init

    def cuda_run(self):
        self.unmap_vectors(self.err_output, self.err_x, self.err_y)
        self.err_x.devmem.from_device_async(self.err_output.devmem)
        self.err_y.devmem.from_device_async(self.err_output.devmem)

    def ocl_run(self):
        self.unmap_vectors(self.err_output, self.err_x, self.err_y)
        self.device.queue_.copy_buffer(
            self.err_output.devmem, self.err_x.devmem, 0, 0,
            self.err_output.nbytes, need_event=False)
        self.device.queue_.copy_buffer(
            self.err_output.devmem, self.err_y.devmem, 0, 0,
            self.err_output.nbytes, need_event=False)

    def numpy_run(self):
        self.err_output.map_read()
        self.err_x.map_invalidate()
        self.err_y.map_invalidate()
        self.err_x.mem[:] = self.err_output.mem[:]
        self.err_y.mem[:] = self.err_output.mem[:]
