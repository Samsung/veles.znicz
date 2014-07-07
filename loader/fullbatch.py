"""
Created on Aug 14, 2013

FullBatchLoader class.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from __future__ import division
import numpy
import opencl4py as cl
from zope.interface import implementer, Interface

import veles.config as config
import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.znicz.loader import (ILoader, Loader)


class IFullBatchLoader(Interface):
    def load_data():
        """Load the data here.
        """


@implementer(ILoader)
class FullBatchLoader(Loader):
    """Loads data entire in memory.

    Attributes:
        original_data: original data (Vector).
        original_labels: original labels (Vector, dtype=numpy.int32)
                         (in case of classification).
        original_target: original target (Vector)
                         (in case of MSE).
        on_device: True to load all data to the device memory.

    Should be overriden in child class:
        load_data()
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchLoader, self).__init__(workflow, **kwargs)
        self.verify_interface(IFullBatchLoader)
        self.on_device = kwargs.get("on_device", False)

    def init_unpickled(self):
        super(FullBatchLoader, self).init_unpickled()
        self.original_data = formats.Vector()
        self.original_labels = formats.Vector()
        self.original_target = formats.Vector()
        self.cl_sources_["fullbatch_loader.cl"] = {}
        self._kernel_target_ = None
        self._global_size = None
        self._global_size_target = None
        self._krn_const = numpy.zeros(2, dtype=numpy.int32)

    def __getstate__(self):
        state = super(FullBatchLoader, self).__getstate__()
        state["original_data"] = None
        state["original_labels"] = None
        state["original_target"] = None
        return state

    def create_minibatches(self):
        self._check_types()

        self.minibatch_data.reset()
        sh = [self.max_minibatch_size]
        sh.extend(self.original_data[0].shape)
        self.minibatch_data.mem = numpy.zeros(
            sh, dtype=opencl_types.dtypes[config.root.common.precision_type])

        self.minibatch_targets.reset()
        if self.original_target.mem is not None:
            sh = [self.max_minibatch_size]
            sh.extend(self.original_target[0].shape)
            self.minibatch_targets.mem = numpy.zeros(
                sh,
                dtype=opencl_types.dtypes[config.root.common.precision_type])

        self.minibatch_labels.reset()
        if self.original_labels.mem is not None:
            sh = [self.max_minibatch_size]
            self.minibatch_labels.mem = numpy.zeros(sh, dtype=numpy.int32)

        self.minibatch_indices.reset()
        self.minibatch_indices.mem = numpy.zeros(self.max_minibatch_size,
                                                 dtype=numpy.int32)

    def _check_types(self):
        if (not isinstance(self.original_data, formats.Vector) or
                not isinstance(self.original_labels, formats.Vector) or
                not isinstance(self.original_target, formats.Vector)):
            raise error.BadFormatError(
                "original_data, original_labels, original_target "
                "should be of type Vector")
        if (self.original_labels.mem is not None and
                self.original_labels.dtype != numpy.int32):
            raise error.BadFormatError(
                "original_labels should have dtype=numpy.int32")

    def initialize(self, device, **kwargs):
        super(FullBatchLoader, self).initialize(device, **kwargs)
        self._check_types()
        if not self.on_device or self.device is None:
            return

        self.info("Will load entire dataset on device")

        self.original_data.initialize(self.device)
        self.minibatch_data.initialize(self.device)
        if self.original_labels:
            self.original_labels.initialize(self.device)
            self.minibatch_labels.initialize(self.device)
        if self.original_target:
            self.original_target.initialize(self.device)
            self.minibatch_target.initialize(self.device)

        if self.shuffled_indices.mem is None:
            self.shuffled_indices.mem = numpy.arange(
                self.total_samples, dtype=numpy.int32)
        self.shuffled_indices.initialize(self.device)
        self.minibatch_indices.initialize(self.device)

        defines = {
            "LABELS": int(self.original_labels.mem is not None),
            "TARGET": int(self.original_target.mem is not None),
            "SAMPLE_SIZE": self.original_data.sample_size,
            "original_data_dtype": opencl_types.numpy_dtype_to_opencl(
                self.original_data.dtype),
            "minibatch_data_dtype": opencl_types.numpy_dtype_to_opencl(
                self.minibatch_data.dtype)
        }
        if self.original_target.mem is not None:
            defines.update({
                "TARGET_SIZE": self.original_target.sample_size,
                "original_target_dtype": opencl_types.numpy_dtype_to_opencl(
                    self.original_data.dtype),
                "minibatch_target_dtype": opencl_types.numpy_dtype_to_opencl(
                    self.minibatch_data.dtype)
            })
        self.build_program(defines, "fullbatch_loader.cl",
                           dtype=self.minibatch_data.dtype)

        self.assign_kernel("fill_minibatch_data_labels")
        if self.original_labels.mem is None:
            self.set_args(self.original_data, self.minibatch_data, cl.skip(2),
                          self.shuffled_indices, self.minibatch_indices)
        else:
            self.set_args(self.original_data, self.minibatch_data, cl.skip(2),
                          self.original_labels, self.minibatch_labels,
                          self.shuffled_indices, self.minibatch_indices)
        self._global_size = [self.max_minibatch_size,
                             self.minibatch_data.sample_size]

        if self.original_target:
            self._kernel_target_ = self.get_kernel("fill_minibatch_target")
            self._kernel_target_.set_args(
                self.original_target.devmem, self.minibatch_target.devmem,
                cl.skip(2), self.shuffled_indices.devmem)
            self._global_size_target = [self.max_minibatch_size,
                                        self.minibatch_target.sample_size]

    def fill_indices(self, start_offset, count):
        if not self.on_device or self.device is None:
            return super(FullBatchLoader, self).fill_indices(start_offset,
                                                             count)
        self.original_data.unmap()
        self.minibatch_data.unmap()

        if self.original_labels.mem is not None:
            self.original_labels.unmap()
            self.minibatch_labels.unmap()

        if self.original_target.mem is not None:
            self.original_target.unmap()
            self.minibatch_target.unmap()

        self.shuffled_indices.unmap()
        self.minibatch_indices.unmap()

        self._krn_const[0] = start_offset
        self._krn_const[1] = count

        self._kernel_.set_arg(2, self._krn_const[0:1])
        self._kernel_.set_arg(3, self._krn_const[1:2])
        self.execute_kernel(self._global_size, None)

        if self.original_target.mem is not None:
            self._kernel_target_.set_arg(2, self._krn_const[0:1])
            self._kernel_target_.set_arg(3, self._krn_const[1:2])
            self.execute_kernel(self._global_size_target, None,
                                self._kernel_target_)

        # No further processing needed, so return True
        return True

    def fill_minibatch(self):
        idxs = self.minibatch_indices.mem

        for i, ii in enumerate(idxs[:self.minibatch_size]):
            self.minibatch_data[i] = self.original_data[int(ii)]

        if self.original_labels.mem is not None:
            for i, ii in enumerate(idxs[:self.minibatch_size]):
                self.minibatch_labels[i] = self.original_labels[int(ii)]

        if self.original_target.mem is not None:
            for i, ii in enumerate(idxs[:self.minibatch_size]):
                self.minibatch_targets[i] = self.original_target[int(ii)]
