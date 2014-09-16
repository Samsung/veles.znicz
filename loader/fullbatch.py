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
from veles.znicz.loader import (ILoader, Loader, LoaderMSE)


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
        self.cl_sources_["fullbatch_loader.cl"] = {}
        self._global_size = None
        self._krn_const = numpy.zeros(2, dtype=numpy.int32)

    def __getstate__(self):
        state = super(FullBatchLoader, self).__getstate__()
        state["original_data"] = None
        state["original_labels"] = None
        return state

    def create_minibatches(self):
        self.check_types()

        self.minibatch_data.reset()
        sh = [self.max_minibatch_size]
        sh.extend(self.original_data[0].shape)
        self.minibatch_data.mem = numpy.zeros(
            sh, dtype=opencl_types.dtypes[config.root.common.precision_type])

        self.minibatch_labels.reset()
        if self.original_labels.mem is not None:
            sh = [self.max_minibatch_size]
            self.minibatch_labels.mem = numpy.zeros(sh, dtype=numpy.int32)

        self.minibatch_indices.reset()
        self.minibatch_indices.mem = numpy.zeros(self.max_minibatch_size,
                                                 dtype=numpy.int32)

    def check_types(self):
        if (not isinstance(self.original_data, formats.Vector) or
                not isinstance(self.original_labels, formats.Vector)):
            raise error.BadFormatError(
                "original_data, original_labels must be of type Vector")
        if (self.original_labels.mem is not None and
                self.original_labels.dtype != numpy.int32):
            raise error.BadFormatError(
                "original_labels should have dtype=numpy.int32")

    def get_ocl_defines(self):
        """Add definitions before building the kernel during initialize().
        """
        return {}

    def initialize(self, device, **kwargs):
        super(FullBatchLoader, self).initialize(device, **kwargs)
        self.check_types()
        if not self.on_device or self.device is None:
            return

        self.info("Will load entire dataset on device")

        self.original_data.initialize(self.device)
        self.minibatch_data.initialize(self.device)
        if self.original_labels:
            self.original_labels.initialize(self.device)
            self.minibatch_labels.initialize(self.device)

        if not self.shuffled_indices:
            self.shuffled_indices.mem = numpy.arange(
                self.total_samples, dtype=numpy.int32)
        self.shuffled_indices.initialize(self.device)
        self.minibatch_indices.initialize(self.device)

        defines = {
            "LABELS": int(self.original_labels.mem is not None),
            "SAMPLE_SIZE": self.original_data.sample_size,
            "original_data_dtype": opencl_types.numpy_dtype_to_opencl(
                self.original_data.dtype),
            "minibatch_data_dtype": opencl_types.numpy_dtype_to_opencl(
                self.minibatch_data.dtype)
        }
        defines.update(self.get_ocl_defines())

        self.build_program(defines, "fullbatch_loader.cl",
                           dtype=self.minibatch_data.dtype)
        self.assign_kernel("fill_minibatch_data_labels")

        if not self.original_labels:
            self._set_args(self.original_data, self.minibatch_data, cl.skip(2),
                           self.shuffled_indices, self.minibatch_indices)
        else:
            self._set_args(self.original_data, self.minibatch_data, cl.skip(2),
                           self.original_labels, self.minibatch_labels,
                           self.shuffled_indices, self.minibatch_indices)
        self._global_size = [self.max_minibatch_size,
                             self.minibatch_data.sample_size]

    def fill_indices(self, start_offset, count):
        if not self.on_device or self.device is None:
            return super(FullBatchLoader, self).fill_indices(start_offset,
                                                             count)
        self.original_data.unmap()
        self.minibatch_data.unmap()

        if self.original_labels:
            self.original_labels.unmap()
            self.minibatch_labels.unmap()

        self.shuffled_indices.unmap()
        self.minibatch_indices.unmap()

        self._krn_const[0] = start_offset
        self._krn_const[1] = count
        self._kernel_.set_arg(2, self._krn_const[0:1])
        self._kernel_.set_arg(3, self._krn_const[1:2])
        self.execute_kernel(self._global_size, None)

        self.on_fill_indices(self._krn_const)

        # No further processing needed, so return True
        return True

    def on_fill_indices(self, krn_consts):
        """Called in the end of fill_indices().
        """
        pass

    def fill_minibatch(self):
        idxs = self.minibatch_indices.mem

        for i, ii in enumerate(idxs[:self.minibatch_size]):
            self.minibatch_data[i] = self.original_data[int(ii)]

        if self.original_labels:
            for i, ii in enumerate(idxs[:self.minibatch_size]):
                self.minibatch_labels[i] = self.original_labels[int(ii)]


class FullBatchLoaderMSE(FullBatchLoader, LoaderMSE):
    """FullBatchLoader for MSE workflows.
    Attributes:
        original_targets: original target (Vector).
    """
    def init_unpickled(self):
        super(FullBatchLoaderMSE, self).init_unpickled()
        self.original_targets = formats.Vector()
        self._kernel_target_ = None
        self._global_size_target = None

    def __getstate__(self):
        state = super(FullBatchLoaderMSE, self).__getstate__()
        state["original_targets"] = None
        return state

    def create_minibatches(self):
        super(FullBatchLoaderMSE, self).create_minibatches()
        self.minibatch_targets.reset()
        sh = [self.max_minibatch_size]
        sh.extend(self.original_targets[0].shape)
        self.minibatch_targets.mem = numpy.zeros(
            sh, dtype=opencl_types.dtypes[config.root.common.precision_type])

    def check_types(self):
        super(FullBatchLoaderMSE, self).check_types()
        if not isinstance(self.original_targets, formats.Vector):
            raise error.BadFormatError(
                "original_targets must be of type Vector")

    def get_ocl_defines(self):
        return {
            "TARGET": 1,
            "TARGET_SIZE": self.original_targets.sample_size,
            "original_target_dtype": opencl_types.numpy_dtype_to_opencl(
                self.original_targets.dtype),
            "minibatch_target_dtype": opencl_types.numpy_dtype_to_opencl(
                self.minibatch_targets.dtype)
        }

    def initialize(self, device, **kwargs):
        super(FullBatchLoaderMSE, self).initialize(device, **kwargs)
        if not self.on_device or self.device is None:
            return

        self.original_targets.initialize(self.device)
        self.minibatch_targets.initialize(self.device)

        self._kernel_target_ = self.get_kernel("fill_minibatch_target")
        self._kernel_target_.set_args(
            self.original_targets.devmem, self.minibatch_targets.devmem,
            cl.skip(2), self.shuffled_indices.devmem)
        self._global_size_target = [self.max_minibatch_size,
                                    self.minibatch_targets.sample_size]

    def on_fill_indices(self, krn_consts):
        self.original_targets.unmap()
        self.minibatch_targets.unmap()
        self._kernel_target_.set_arg(2, krn_consts[0:1])
        self._kernel_target_.set_arg(3, krn_consts[1:2])
        self.execute_kernel(self._global_size_target, None,
                            self._kernel_target_)

    def fill_minibatch(self):
        super(FullBatchLoaderMSE, self).fill_minibatch()
        for i, v in enumerate(
                self.minibatch_indices.mem[:self.minibatch_size]):
            self.minibatch_targets[i] = self.original_targets[int(v)]
