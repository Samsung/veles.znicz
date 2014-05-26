"""
Created on Apr 1, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import math

from veles.config import root
import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.opencl_units import OpenCLUnit


class EvaluatorSoftmax(OpenCLUnit):
    """Evaluator for nn softmax output from the batch labels.

    Must be assigned before initialize():
        output
        labels
        batch_size
        max_idx
        max_samples_per_epoch

    Updates after run():
        err_output
        n_err
        confusion_matrix
        max_err_output_sum

    Creates within initialize():
        err_output
        n_err
        confusion_matrix
        max_err_output_sum

    Attributes:
        labels: labels for Batch.
        output: output of the network_common as Batch.
        err_output: backpropagation errors based on labels.
        batch_size: number of elements in output to evaluate.
        max_samples_per_epoch: maximum number of samples per epoch,
            will choose n_err element type based on it.
        confusion_matrix: confusion matrix for the output.
        compute_confusion_matrix: compute confusion matrix or not.
        max_idx: indexes of element with maximum real value for each sample.
        max_err_output_sum: maximum of backpropagated error sum by sample.
        krn_constants_i_: numpy array for constant arguments to kernel.
    """
    def __init__(self, workflow, **kwargs):
        compute_confusion_matrix = kwargs.get("compute_confusion_matrix", True)
        kwargs["compute_confusion_matrix"] = compute_confusion_matrix
        kwargs["view_group"] = kwargs.get("view_group", "EVALUATOR")
        super(EvaluatorSoftmax, self).__init__(workflow, **kwargs)
        self.labels = None  # formats.Vector()
        self.output = None  # formats.Vector()
        self.err_output = formats.Vector()
        self.batch_size = 0
        self.max_samples_per_epoch = 0
        self.compute_confusion_matrix = compute_confusion_matrix
        self.confusion_matrix = formats.Vector()
        self.n_err = formats.Vector()
        self.max_idx = None  # formats.Vector()
        self.krn_constants_i_ = None
        self.max_err_output_sum = formats.Vector()

    def initialize(self, device, **kwargs):
        super(EvaluatorSoftmax, self).initialize(device=device, **kwargs)
        self.cl_sources_["evaluator.cl"] = {}

        if (self.err_output.mem is None or
                self.err_output.mem.size != self.output.mem.size):
            self.err_output.reset()
            self.err_output.mem = numpy.zeros(self.output.mem.shape,
                                              dtype=self.output.mem.dtype)

        if self.n_err.mem is None or self.n_err.mem.size < 2:
            self.n_err.reset()
            self.n_err.mem = numpy.zeros(2, dtype=numpy.int32)

        out_size = self.output.mem.size // self.output.mem.shape[0]
        if self.compute_confusion_matrix:
            if (self.confusion_matrix.mem is None or
                    self.confusion_matrix.mem.size != out_size * out_size):
                self.confusion_matrix.reset()
                self.confusion_matrix.mem = numpy.zeros(
                    [out_size, out_size], dtype=numpy.int32)
        else:
            self.confusion_matrix.reset()

        if (self.max_err_output_sum.mem is None or
                self.max_err_output_sum.mem.size < 1):
            self.max_err_output_sum.reset()
            self.max_err_output_sum.mem = numpy.zeros(
                1, dtype=opencl_types.dtypes[root.common.dtype])

        self.output.initialize(self.device)
        self.err_output.initialize(self.device)
        self.confusion_matrix.initialize(self.device)
        self.n_err.initialize(self.device)
        self.max_idx.initialize(self.device)
        self.labels.initialize(self.device)
        self.max_err_output_sum.initialize(self.device)

        if self.device is None:
            return

        self.krn_constants_i_ = numpy.zeros(1, dtype=numpy.int32)

        if self.program_ is None:
            defines = {
                'BLOCK_SIZE':
                self.device.device_info.BLOCK_SIZE[
                    opencl_types.numpy_dtype_to_opencl(self.output.mem.dtype)],
                'BATCH': self.err_output.mem.shape[0],
                'Y': self.err_output.mem.size // self.err_output.mem.shape[0],
            }

            self.build_program(defines, "ev_%d.cl" %
                               (self.output.mem.size //
                                self.output.mem.shape[0]),
                               dtype=self.output.mem.dtype)

            self.assign_kernel("ev_sm")
            self.set_args(self.output, self.max_idx, self.labels,
                          self.err_output, self.n_err, self.confusion_matrix,
                          self.max_err_output_sum)

    def ocl_run(self):
        self.err_output.unmap()
        self.output.unmap()
        self.max_idx.unmap()
        self.labels.unmap()
        self.n_err.unmap()
        self.confusion_matrix.unmap()
        self.max_err_output_sum.unmap()

        self.krn_constants_i_[0] = self.batch_size
        self.set_arg(7, self.krn_constants_i_[0:1])

        local_size = [self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.output.mem.dtype)]]
        global_size = [local_size[0]]
        event = self.execute_kernel(global_size, local_size)
        event.wait()

    def cpu_run(self):
        self.err_output.map_invalidate()
        self.output.map_read()
        self.max_idx.map_read()
        self.labels.map_read()
        self.n_err.map_write()
        self.confusion_matrix.map_write()
        self.max_err_output_sum.map_write()

        batch_size = self.batch_size
        labels = self.labels.mem
        confusion_matrix = self.confusion_matrix.mem

        n_ok = 0
        for i in range(batch_size):  # loop by batch
            output = formats.ravel(self.output[i])
            err_output = formats.ravel(self.err_output[i])

            max_idx = self.max_idx[i]
            confusion_matrix[max_idx, labels[i]] += 1
            if max_idx == labels[i]:
                n_ok += 1

            # Compute softmax output error gradient
            err_output[:] = output[:]
            err_output[labels[i]] -= 1.0
            if err_output.dtype in (numpy.complex64, numpy.complex128):
                self.max_err_output_sum[0] = max(
                    self.max_err_output_sum[0], numpy.linalg.norm(err_output))
            else:
                self.max_err_output_sum[0] = max(
                    self.max_err_output_sum[0], (numpy.fabs(err_output)).sum())
        # Set errors for excessive samples to zero
        if batch_size < self.err_output.mem.shape[0]:
            self.err_output.mem[batch_size:] = 0.0
        self.n_err[0] += batch_size - n_ok


class EvaluatorMSE(OpenCLUnit):
    """Evaluator for nn softmax output from the batch labels.

    Must be assigned before initialize():
        output
        target
        batch_size
        max_samples_per_epoch
        labels (may be None)
        class_target (may be None)

    Updates after run():
        err_output
        confusion_matrix
        max_err_output_sum
        n_err (only if labels and class_target is not None)

    Creates within initialize():
        err_output
        n_err (only if labels and class_target is not None)
        max_err_output_sum

    Attributes:
        output: output of the network_common as Batch.
        target: target for the current Batch.
        err_output: backpropagation errors.
        batch_size: number of elements in output to evaluate.
        metrics: [0] - sum of sample's mse, [1] - max of sample's mse,
                 [2] - min of sample's mse.
        mse: array of mse for each sample in minibatch.
        krn_constants_i_: numpy array for constant arguments to kernel.
        labels: labels for a Batch (may be None).
        class_target: target for each class (may be None).
        n_err: number of wrong recognized samples
            (if labels and class_target is not None).
        max_samples_per_epoch: maximum number of samples per epoch,
            will choose n_err element type based on it.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "EVALUATOR")
        super(EvaluatorMSE, self).__init__(workflow, **kwargs)
        self.output = None  # formats.Vector()
        self.target = None  # formats.Vector()
        self.err_output = formats.Vector()
        self.batch_size = None  # [0]
        self.krn_constants_i_ = None
        self.metrics = formats.Vector()
        self.mse = formats.Vector()
        self.labels = None
        self.class_target = None
        self.max_samples_per_epoch = None  # [0]
        self.n_err = formats.Vector()

    def initialize(self, device, **kwargs):
        super(EvaluatorMSE, self).initialize(device=device, **kwargs)
        self.cl_sources_["evaluator.cl"] = {}

        if (self.err_output.mem is None or
                self.err_output.mem.size != self.output.mem.size):
            self.err_output.reset()
            self.err_output.mem = numpy.zeros(self.output.mem.shape,
                                              dtype=self.output.mem.dtype)

        if self.metrics.mem is None or self.metrics.mem.size < 3:
            self.metrics.reset()
            self.metrics.mem = numpy.zeros(
                3, dtype=opencl_types.dtypes[root.common.dtype])
            self.metrics[2] = 1.0e30  # mse_min

        if (self.mse.mem is None or
                self.mse.mem.size != self.err_output.mem.shape[0]):
            self.mse.reset()
            self.mse.mem = numpy.zeros(
                self.err_output.mem.shape[0],
                dtype=opencl_types.dtypes[root.common.dtype])

        if (self.labels is not None and self.class_target is not None and
                (self.n_err.mem is None or self.n_err.mem.size < 2)):
            self.n_err.reset()
            self.n_err.mem = numpy.zeros(2, dtype=numpy.int32)
            self.cl_sources_["mse_find_closest.cl"] = {}
            self.class_target.initialize(self.device)
            self.labels.initialize(self.device)
            self.n_err.initialize(self.device)

        self.output.initialize(self.device)
        self.err_output.initialize(self.device)
        self.target.initialize(self.device)
        self.metrics.initialize(self.device)
        self.mse.initialize(self.device)

        if not self.device:
            return

        self.krn_constants_i_ = numpy.zeros(1, dtype=numpy.int32)

        if self.program_ is None:
            defines = {
                'BLOCK_SIZE':
                self.device.device_info.BLOCK_SIZE[
                    opencl_types.numpy_dtype_to_opencl(self.output.mem.dtype)],
                'BATCH': self.err_output.mem.shape[0],
                'Y': self.err_output.mem.size // self.err_output.mem.shape[0],
                'SAMPLE_SIZE': 'Y',
                'N_TARGETS': (self.class_target.mem.shape[0]
                              if self.class_target is not None else 0)}

            self.build_program(defines, "ev_%d.cl" %
                               (self.output.mem.size //
                                self.output.mem.shape[0]),
                               dtype=self.output.mem.dtype)

            self.assign_kernel("ev_mse")
            self.set_args(self.output, self.target, self.err_output,
                          self.metrics, self.mse.devmem)

            if self.labels is not None and self.class_target is not None:
                self.krn_find_closest_ = self.get_kernel("mse_find_closest")
                self.krn_find_closest_.set_args(
                    self.output.devmem,
                    self.class_target.devmem,
                    self.labels.devmem,
                    self.n_err.devmem)

    def ocl_run(self):
        self.err_output.unmap()
        self.output.unmap()
        self.target.unmap()
        self.metrics.unmap()
        self.mse.unmap()

        batch_size = self.batch_size
        self.krn_constants_i_[0] = batch_size
        self.set_arg(5, self.krn_constants_i_[0:1])

        local_size = [self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.output.mem.dtype)]]
        global_size = [local_size[0]]
        event = self.execute_kernel(global_size, local_size)
        event.wait()

        # Do the following part on CPU (GPU version not implemented currently)
        if self.labels is not None and self.class_target is not None:
            self.class_target.unmap()
            self.labels.unmap()
            self.n_err.unmap()
            self.execute_kernel([batch_size], None,
                                self.krn_find_closest_).wait()

    def cpu_run(self):
        self.output.map_read()
        self.target.map_read()
        self.metrics.map_write()
        self.err_output.map_invalidate()
        self.mse.map_invalidate()

        assert(self.output.mem.size == self.target.mem.size ==
               self.err_output.mem.size)
        for i in range(self.err_output.mem.shape[0]):
            if i < self.batch_size:
                it = numpy.nditer([self.output.mem[i], self.target.mem[i],
                                   self.err_output.mem[i], self.mse.mem[i]],
                                  op_flags=[['readonly'], ['readonly'],
                                            ['writeonly'], ['writeonly']])
                sum_err2 = 0
                counter = 0
                for y, t, err_y in it:
                    err_y[...] = y - t
                    sum_err2 += err_y * err_y
                    counter += 1
                self.mse.mem[i] = math.sqrt(sum_err2 / counter)
            else:
                self.err_output.mem[i] = 0
                self.mse.mem[i] = 0
        self.metrics.mem[0] += numpy.sum(self.mse.mem)
        self.metrics.mem[1] = max(self.metrics.mem[1], self.mse.mem.max())
        self.metrics.mem[2] = min(self.metrics.mem[2], self.mse.mem.min())
