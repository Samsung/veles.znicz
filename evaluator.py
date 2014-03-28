"""
Created on Apr 1, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy

from veles.config import root
import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.units as units


class EvaluatorSoftmax(units.OpenCLUnit):
    """Evaluator for nn softmax output from the batch labels.

    Should be assigned before initialize():
        y
        labels
        batch_size
        max_idx
        max_samples_per_epoch

    Updates after run():
        err_y
        n_err
        confusion_matrix
        max_err_y_sum

    Creates within initialize():
        err_y
        n_err
        confusion_matrix
        max_err_y_sum

    Attributes:
        labels: labels for Batch.
        y: output of the network_common as Batch.
        err_y: backpropagation errors based on labels.
        batch_size: number of elements in y to evaluate.
        max_samples_per_epoch: maximum number of samples per epoch,
            will choose n_err element type based on it.
        confusion_matrix: confusion matrix for the output.
        compute_confusion_matrix: compute confusion matrix or not.
        max_idx: indexes of element with maximum real value for each sample.
        max_err_y_sum: maximum of backpropagated error sum by sample.
        krn_constants_i_: numpy array for constant arguments to kernel.
    """
    def __init__(self, workflow, **kwargs):
        compute_confusion_matrix = kwargs.get("compute_confusion_matrix", True)
        kwargs["compute_confusion_matrix"] = compute_confusion_matrix
        kwargs["view_group"] = kwargs.get("view_group", "EVALUATOR")
        super(EvaluatorSoftmax, self).__init__(workflow, **kwargs)
        self.labels = None  # formats.Vector()
        self.y = None  # formats.Vector()
        self.err_y = formats.Vector()
        self.batch_size = None  # [0]
        self.max_samples_per_epoch = None  # [0]
        self.compute_confusion_matrix = compute_confusion_matrix
        self.confusion_matrix = formats.Vector()
        self.n_err = formats.Vector()
        self.max_idx = None  # formats.Vector()
        self.krn_constants_i_ = None
        self.max_err_y_sum = formats.Vector()

    def initialize(self):
        super(EvaluatorSoftmax, self).initialize()
        itype = opencl_types.get_itype_from_size(
            self.y.v.size // self.y.v.shape[0])
        if (self.labels.v.dtype != opencl_types.itypes[itype] or
                self.labels.v.dtype != self.max_idx.v.dtype):
            raise error.ErrBadFormat("Incorrectly set labels.dtype "
                                     "(probably in Loader).")
        itype2 = opencl_types.get_itype_from_size(
            self.max_samples_per_epoch[0])
        self.cl_sources_["evaluator.cl"] = {"itype": itype, "itype2": itype2}

        if (self.err_y.v is None or
                self.err_y.v.size != self.y.v.size):
            self.err_y.reset()
            self.err_y.v = numpy.zeros(self.y.v.shape, dtype=self.y.v.dtype)

        if self.n_err.v is None or self.n_err.v.size < 2:
            self.n_err.reset()
            self.n_err.v = numpy.zeros(2, dtype=opencl_types.itypes[itype2])

        out_size = self.y.v.size // self.y.v.shape[0]
        if self.compute_confusion_matrix:
            if (self.confusion_matrix.v is None or
                    self.confusion_matrix.v.size != out_size * out_size):
                self.confusion_matrix.reset()
                self.confusion_matrix.v = numpy.zeros(
                    [out_size, out_size], dtype=opencl_types.itypes[itype2])
        else:
            self.confusion_matrix.reset()

        if self.max_err_y_sum.v is None or self.max_err_y_sum.v.size < 1:
            self.max_err_y_sum.reset()
            self.max_err_y_sum.v = numpy.zeros(
                1, dtype=opencl_types.dtypes[root.common.dtype])

        self.y.initialize(self.device)
        self.err_y.initialize(self.device)
        self.confusion_matrix.initialize(self.device)
        self.n_err.initialize(self.device)
        self.max_idx.initialize(self.device)
        self.labels.initialize(self.device)
        self.max_err_y_sum.initialize(self.device)

        if self.device is None:
            return

        self.krn_constants_i_ = numpy.zeros(1, opencl_types.itypes[itype2])

        if self.prg_ is None:
            defines = {
                'BLOCK_SIZE':
                self.device.device_info.BLOCK_SIZE[root.common.precision_type],
                'BATCH': self.err_y.v.shape[0],
                'Y': self.err_y.v.size // self.err_y.v.shape[0],
            }
            self.build_program(defines, "%s/ev_%d.cl" %
                               (root.common.cache_dir,
                                self.y.v.size // self.y.v.shape[0]))

            self.krn_ = self.get_kernel("ev_sm")
            self.krn_.set_arg(0, self.y.v_)
            self.krn_.set_arg(1, self.max_idx.v_)
            self.krn_.set_arg(2, self.labels.v_)
            self.krn_.set_arg(3, self.err_y.v_)
            self.krn_.set_arg(4, self.n_err.v_)
            self.krn_.set_arg(5, self.confusion_matrix.v_)
            self.krn_.set_arg(6, self.max_err_y_sum.v_)

    def gpu_run(self):
        self.err_y.unmap()
        self.y.unmap()
        self.max_idx.unmap()
        self.labels.unmap()
        self.n_err.unmap()
        self.confusion_matrix.unmap()
        self.max_err_y_sum.unmap()

        self.krn_constants_i_[0] = self.batch_size[0]
        self.krn_.set_arg(7, self.krn_constants_i_[0:1])

        local_size = [self.device.device_info.BLOCK_SIZE[
            root.common.precision_type]]
        global_size = [local_size[0]]
        event = self.execute_kernel(self.krn_, global_size, local_size)
        event.wait()

    def cpu_run(self):
        self.err_y.map_invalidate()
        self.y.map_read()
        self.max_idx.map_read()
        self.labels.map_read()
        self.n_err.map_write()
        self.confusion_matrix.map_write()
        self.max_err_y_sum.map_write()

        batch_size = self.batch_size[0]
        labels = self.labels.v
        confusion_matrix = self.confusion_matrix.v

        n_ok = 0
        for i in range(batch_size):  # loop by batch
            y = formats.ravel(self.y.v[i])
            err_y = formats.ravel(self.err_y.v[i])

            max_idx = self.max_idx.v[i]
            confusion_matrix[max_idx, labels[i]] += 1
            if max_idx == labels[i]:
                n_ok += 1

            # Compute softmax output error gradient
            err_y[:] = y[:]
            err_y[labels[i]] -= 1.0
            if err_y.dtype in (numpy.complex64, numpy.complex128):
                self.max_err_y_sum.v[0] = max(self.max_err_y_sum.v[0],
                                              numpy.linalg.norm(err_y))
            else:
                self.max_err_y_sum.v[0] = max(self.max_err_y_sum.v[0],
                                              (numpy.fabs(err_y)).sum())
        # Set errors for excessive samples to zero
        if batch_size < self.err_y.v.shape[0]:
            self.err_y.v[batch_size:] = 0.0
        self.n_err.v[0] += batch_size - n_ok


class EvaluatorMSE(units.OpenCLUnit):
    """Evaluator for nn softmax output from the batch labels.

    Should be assigned before initialize():
        y
        target
        batch_size
        max_samples_per_epoch
        labels (may be None)
        class_target (may be None)

    Updates after run():
        err_y
        confusion_matrix
        max_err_y_sum
        n_err (only if labels and class_target is not None)

    Creates within initialize():
        err_y
        n_err (only if labels and class_target is not None)
        max_err_y_sum

    Attributes:
        y: output of the network_common as Batch.
        target: target for the current Batch.
        err_y: backpropagation errors.
        batch_size: number of elements in y to evaluate.
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
        self.y = None  # formats.Vector()
        self.target = None  # formats.Vector()
        self.err_y = formats.Vector()
        self.batch_size = None  # [0]
        self.krn_constants_i_ = None
        self.metrics = formats.Vector()
        self.mse = formats.Vector()
        self.labels = None
        self.class_target = None
        self.max_samples_per_epoch = None  # [0]
        self.n_err = formats.Vector()

    def initialize(self):
        super(EvaluatorMSE, self).initialize()
        itype = opencl_types.get_itype_from_size(
            (self.y.v.size // self.y.v.shape[0]))
        itype2 = opencl_types.get_itype_from_size(
            self.max_samples_per_epoch[0])
        self.cl_sources_["evaluator.cl"] = {"itype": itype, "itype2": itype2}

        if (self.err_y.v is None or
                self.err_y.v.size != self.y.v.size):
            self.err_y.reset()
            self.err_y.v = numpy.zeros(self.y.v.shape, dtype=self.y.v.dtype)

        if self.metrics.v is None or self.metrics.v.size < 3:
            self.metrics.reset()
            self.metrics.v = numpy.zeros(
                3, dtype=opencl_types.dtypes[root.common.dtype])
            self.metrics.v[2] = 1.0e30  # mse_min

        if (self.mse.v is None or
                self.mse.v.size != self.err_y.v.shape[0]):
            self.mse.reset()
            self.mse.v = numpy.zeros(
                self.err_y.v.shape[0],
                dtype=opencl_types.dtypes[root.common.dtype])

        if (self.labels is not None and self.class_target is not None and
                (self.n_err.v is None or self.n_err.v.size < 2)):
            itype0 = opencl_types.get_itype_from_size(len(self.class_target.v))
            if self.labels.v.dtype != opencl_types.itypes[itype0]:
                raise error.ErrBadFormat("Incorrectly set labels.dtype "
                                         "(probably in Loader).")
            self.n_err.reset()
            self.n_err.v = numpy.zeros(2, dtype=opencl_types.itypes[itype2])
            self.cl_sources_["mse_find_closest.cl"] = {"itype0": itype0}
            self.class_target.initialize(self.device)
            self.labels.initialize(self.device)
            self.n_err.initialize(self.device)

        self.y.initialize(self.device)
        self.err_y.initialize(self.device)
        self.target.initialize(self.device)
        self.metrics.initialize(self.device)
        self.mse.initialize(self.device)

        if not self.device:
            return

        self.krn_constants_i_ = numpy.zeros(1, opencl_types.itypes[itype2])

        if self.prg_ is None:
            defines = {
                'BLOCK_SIZE':
                self.device.device_info.BLOCK_SIZE[root.common.precision_type],
                'BATCH': self.err_y.v.shape[0],
                'Y': self.err_y.v.size // self.err_y.v.shape[0],
                'SAMPLE_SIZE': 'Y',
                'N_TARGETS': (self.class_target.v.shape[0]
                              if self.class_target is not None else 0)
            }
            self.build_program(
                defines, "%s/ev_%d.cl" % (root.common.cache_dir,
                                          self.y.v.size // self.y.v.shape[0]))

            self.krn_ = self.get_kernel("ev_mse")
            self.krn_.set_arg(0, self.y.v_)
            self.krn_.set_arg(1, self.target.v_)
            self.krn_.set_arg(2, self.err_y.v_)
            self.krn_.set_arg(3, self.metrics.v_)
            self.krn_.set_arg(4, self.mse.v_)

            if self.labels is not None and self.class_target is not None:
                self.krn_find_closest_ = self.get_kernel("mse_find_closest")
                self.krn_find_closest_.set_arg(0, self.y.v_)
                self.krn_find_closest_.set_arg(1, self.class_target.v_)
                self.krn_find_closest_.set_arg(2, self.labels.v_)
                self.krn_find_closest_.set_arg(3, self.n_err.v_)

    def gpu_run(self):
        self.err_y.unmap()
        self.y.unmap()
        self.target.unmap()
        self.metrics.unmap()
        self.mse.unmap()

        batch_size = self.batch_size[0]
        self.krn_constants_i_[0] = batch_size
        self.krn_.set_arg(5, self.krn_constants_i_[0:1])

        local_size = [self.device.device_info.BLOCK_SIZE[
            root.common.precision_type]]
        global_size = [local_size[0]]
        event = self.execute_kernel(self.krn_, global_size, local_size)
        event.wait()

        # Do the following part on CPU (GPU version not implemented currently)
        if self.labels is not None and self.class_target is not None:
            self.class_target.unmap()
            self.labels.unmap()
            self.n_err.unmap()
            event = self.execute_kernel(self.krn_find_closest_,
                                        [batch_size], None)
            event.wait()

    def cpu_run(self):
        raise error.ErrNotImplemented()
