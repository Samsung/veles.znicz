"""
Created on Apr 1, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import formats
import numpy
import time
import config
import pyopencl
import error


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
        n_err_skipped
        confusion_matrix
        skipped
        max_err_y_sum

    Creates within initialize():
        err_y
        n_err_skipped
        confusion_matrix
        skipped
        max_err_y_sum

    Attributes:
        labels: labels for Batch.
        y: output of the network as Batch.
        err_y: backpropagation errors based on labels.
        threshold: when output becomes greater than this value,
            assume gradient as 0.
        threshold_low: when gradient was assumed as 0 and output becomes less
            than this value, calculate gradient as usual.
        skipped: array of bytes with non-zero value if the sample was skipped
            due to assumed zero-gradient.
        batch_size: number of elements in y to evaluate.
        max_samples_per_epoch: maximum number of samples per epoch,
            will choose n_err_skipped element type based on it.
        confusion_matrix: confusion matrix for the output.
        compute_confusion_matrix: compute confusion matrix or not.
        max_idx: indexes of element with maximum value for each sample.
        max_err_y_sum: maximum of backpropagated error sum by sample.
        krn_constants_d_: numpy array for constant arguments to kernel.
        krn_constants_i_: numpy array for constant arguments to kernel.
    """
    def __init__(self, threshold=1.0, threshold_low=None, device=None,
                 compute_confusion_matrix=True):
        super(EvaluatorSoftmax, self).__init__(device=device)
        self.labels = None  # formats.Vector()
        self.y = None  # formats.Vector()
        self.err_y = formats.Vector()
        self.batch_size = None  # [0]
        self.max_samples_per_epoch = None  # [0]
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.skipped = formats.Vector()
        self.n_skipped = None
        self.compute_confusion_matrix = compute_confusion_matrix
        self.confusion_matrix = formats.Vector()
        self.n_err_skipped = formats.Vector()
        self.max_idx = None  # formats.Vector()
        self.krn_constants_d_ = None
        self.krn_constants_i_ = None
        self.max_err_y_sum = formats.Vector()

    def initialize(self):
        itype = config.get_itype_from_size(self.y.v.size // self.y.v.shape[0])
        if (self.labels.v.dtype != config.itypes[itype] or
            self.labels.v.dtype != self.max_idx.v.dtype):
            raise error.ErrBadFormat("Uncorrectly set labels.dtype "
                                     "(probably in Loader).")
        itype2 = config.get_itype_from_size(self.max_samples_per_epoch[0])
        global this_dir
        self.cl_sources_["%s/evaluator.cl" % (config.cl_dir)] = (
            "#define itype %s\n#define itype2 %s" % (itype, itype2))

        if (self.err_y.v == None or
            self.err_y.v.size != self.y.v.size):
            self.err_y.v = numpy.zeros(self.y.v.shape,
                dtype=config.dtypes[config.dtype])
            self.err_y.v_ = None

        if self.n_err_skipped.v == None or self.n_err_skipped.v.size < 2:
            self.n_err_skipped.v = numpy.zeros(2, dtype=config.itypes[itype2])
            self.n_err_skipped.v_ = None

        if (self.skipped.v == None or
            self.skipped.v.size != self.y.v.shape[0]):
            self.skipped.v = numpy.zeros(self.y.v.shape[0],
                dtype=numpy.int8)
            self.skipped.v_ = None

        out_size = self.y.v.size // self.y.v.shape[0]
        if self.compute_confusion_matrix:
            if (self.confusion_matrix.v == None or
                self.confusion_matrix.v.size != out_size * out_size):
                self.confusion_matrix.v = numpy.zeros([out_size, out_size],
                    dtype=config.itypes[itype2])
                self.confusion_matrix.v_ = None
        else:
            self.confusion_matrix.v = None
            self.confusion_matrix.v_ = None
            self.confusion_matrix.aligned_ = None

        if self.max_err_y_sum.v == None or self.max_err_y_sum.v.size < 1:
            self.max_err_y_sum.v = numpy.zeros(1,
                dtype=config.dtypes[config.dtype])
            self.max_err_y_sum.v_ = None

        self.err_y.initialize(self.device)
        self.confusion_matrix.initialize(self.device)
        self.skipped.initialize(self.device)
        self.n_err_skipped.initialize(self.device)
        self.max_idx.initialize(self.device)
        self.labels.initialize(self.device)
        self.max_err_y_sum.initialize(self.device)

        if self.device == None:
            return

        self.krn_constants_d_ = numpy.zeros(2, config.dtypes[config.dtype])
        self.krn_constants_i_ = numpy.zeros(1, config.itypes[itype2])

        if self.prg_ == None:
            defines = ("%s\n"
                       "#define BLOCK_SIZE %d\n"
                       "#define BATCH %d\n"
                       "#define Y %d\n"
                       "#define Y_REAL %d\n\n") % \
                   (config.cl_defines[config.dtype],
                    self.device.info.BLOCK_SIZE[config.dtype],
                    self.err_y.aligned_.shape[0],
                    self.err_y.aligned_.size // self.err_y.aligned_.shape[0],
                    self.err_y.v.size // self.err_y.v.shape[0])
            s = defines
            for src, define in self.cl_sources_.items():
                s += "\n" + define + "\n"
                fin = open(src, "r")
                s += fin.read()
                fin.close()
            fout = open("%s/ev_%d.cl" % (config.cache_dir,
                self.y.v.size // self.y.v.shape[0]), "w")
            fout.write(s)
            fout.close()

            self.prg_ = pyopencl.Program(self.device.context_, s).build()

            self.krn_ = pyopencl.Kernel(self.prg_, "ev_sm")
            self.krn_.set_arg(0, self.y.v_)
            self.krn_.set_arg(1, self.max_idx.v_)
            self.krn_.set_arg(2, self.labels.v_)
            self.krn_.set_arg(3, self.err_y.v_)
            self.krn_.set_arg(4, self.skipped.v_)
            self.krn_.set_arg(5, self.n_err_skipped.v_)
            self.krn_.set_arg(6, self.confusion_matrix.v_)
            self.krn_.set_arg(7, self.max_err_y_sum.v_)

    def gpu_run(self):
        return self.cpu_run()
        t1 = time.time()

        threshold = self.threshold
        threshold_low = self.threshold_low
        if threshold_low == None:
            threshold_low = threshold
        batch_size = self.batch_size[0]

        self.y.sync(formats.GPU)
        self.max_idx.sync(formats.GPU)
        self.labels.sync(formats.GPU)
        self.skipped.sync(formats.GPU)
        self.n_err_skipped.sync(formats.GPU)
        self.confusion_matrix.sync(formats.GPU)
        self.max_err_y_sum.sync(formats.GPU)

        self.krn_constants_i_[0] = batch_size
        self.krn_constants_d_[0] = threshold
        self.krn_constants_d_[1] = threshold_low
        self.krn_.set_arg(8, self.krn_constants_i_[0])
        self.krn_.set_arg(9, self.krn_constants_d_[0])
        self.krn_.set_arg(10, self.krn_constants_d_[1])

        local_size = [self.device.info.BLOCK_SIZE[config.dtype]]
        global_size = [local_size[0]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
            self.krn_, global_size, local_size)
        event.wait()

        self.err_y.update(formats.GPU)
        self.confusion_matrix.update(formats.GPU)
        self.skipped.update(formats.GPU)
        self.n_err_skipped.update(formats.GPU)
        self.max_err_y_sum.update(formats.GPU)

        self.log().debug("%s in %.2f sec" % (self.__class__.__name__,
                                             time.time() - t1))

    def cpu_run(self):
        t1 = time.time()
        batch_size = self.batch_size[0]

        self.y.sync()
        self.max_idx.sync()
        self.labels.sync()
        self.skipped.sync()
        self.n_err_skipped.sync()
        self.confusion_matrix.sync()
        self.max_err_y_sum.sync()

        labels = self.labels.v

        threshold = self.threshold
        threshold_low = self.threshold_low
        if threshold_low == None:
            threshold_low = threshold

        confusion_matrix = self.confusion_matrix.v

        n_ok = 0
        n_skip = 0
        for i in range(0, batch_size):  # loop by batch
            y = formats.ravel(self.y.v[i])
            err_y = formats.ravel(self.err_y.v[i])

            skip = False
            max_idx = self.max_idx.v[i]
            max_vle = y[max_idx]
            confusion_matrix[max_idx, labels[i]] += 1
            if max_idx == labels[i]:
                n_ok += 1
                # check for threshold
                if (max_vle > threshold) or \
                   ((max_vle > threshold_low) and (self.skipped[i])):
                    err_y[:] = 0  # already trained good enough, skip it
                    self.skipped.v[i] = 1
                    skip = True
                    n_skip += 1

            if not skip:
                # Compute softmax output error gradient
                err_y[:] = y[:]
                err_y[labels[i]] = y[labels[i]] - 1.0
                self.skipped.v[i] = 0
                self.max_err_y_sum.v[0] = max(self.max_err_y_sum.v[0],
                    numpy.sum(numpy.fabs(err_y)))
        # Set errors for excessive samples to zero
        if batch_size < self.err_y.v.shape[0]:
            self.err_y.v[batch_size:] = 0.0
        self.n_err_skipped.v[0] += batch_size - n_ok
        self.n_err_skipped.v[1] += n_skip

        self.err_y.update()
        self.confusion_matrix.update()
        self.skipped.update()
        self.n_err_skipped.update()
        self.max_err_y_sum.update()
        self.log().debug("%s in %.2f sec" % (self.__class__.__name__,
                                             time.time() - t1))


class EvaluatorMSE(units.OpenCLUnit):
    """Evaluator for nn softmax output from the batch labels.

    Should be assigned before initialize():
        y
        labels
        batch_size
        max_samples_per_epoch
        class_target

    Updates after run():
        err_y
        n_err_skipped
        confusion_matrix
        skipped
        max_err_y_sum

    Creates within initialize():
        err_y
        n_err_skipped
        skipped
        max_err_y_sum

    Attributes:
        labels: labels for Batch.
        y: output of the network as Batch.
        err_y: backpropagation errors based on labels.
        threshold: when difference between output and target becomes lower
            than this value, assume gradient as 0.
        batch_size: number of elements in y to evaluate.
        max_samples_per_epoch: maximum number of samples per epoch,
            will choose n_err_skipped element type based on it.
        metrics: [0] - sse, [1] - maximum of backpropagated error sum.
        mse: array of mse for each sample in minibatch.
        krn_constants_d_: numpy array for constant arguments to kernel.
        krn_constants_i_: numpy array for constant arguments to kernel.
        class_target: target for each class (may be None).
    """
    def __init__(self, device=None, threshold_skip=0.0, threshold_ok=0.0):
        super(EvaluatorMSE, self).__init__(device=device)
        self.target = None  # formats.Vector()
        self.y = None  # formats.Vector()
        self.err_y = formats.Vector()
        self.batch_size = None  # [0]
        self.max_samples_per_epoch = None  # [0]
        self.threshold_skip = threshold_skip
        self.threshold_ok = threshold_ok
        self.n_err_skipped = formats.Vector()
        self.krn_constants_d_ = None
        self.krn_constants_i_ = None
        self.metrics = formats.Vector()
        self.effective_batch_size = [0]
        self.mse = formats.Vector()
        self.class_target = None
        self.labels = None

    def initialize(self):
        itype = config.get_itype_from_size((self.y.v.size //
                                            self.y.v.shape[0]))
        itype2 = config.get_itype_from_size(self.max_samples_per_epoch[0])
        self.cl_sources_["%s/evaluator.cl" % (config.cl_dir)] = (
            "#define itype %s\n#define itype2 %s" % (itype, itype2))

        if (self.err_y.v == None or
            self.err_y.v.size != self.y.v.size):
            self.err_y.v = numpy.zeros(self.y.v.shape,
                dtype=config.dtypes[config.dtype])
            self.err_y.v_ = None

        if self.n_err_skipped.v == None or self.n_err_skipped.v.size < 2:
            self.n_err_skipped.v = numpy.zeros(2, dtype=config.itypes[itype2])
            self.n_err_skipped.v_ = None

        if self.metrics.v == None or self.metrics.v.size < 3:
            self.metrics.v = numpy.zeros(3,
                dtype=config.dtypes[config.dtype])
            self.metrics.v[2] = 1.0e30  # mse_min
            self.metrics.v_ = None

        if (self.mse.v == None or
            self.mse.v.size != self.err_y.v.shape[0]):
            self.mse.v = numpy.zeros(self.err_y.v.shape[0],
                dtype=config.dtypes[config.dtype])
            self.mse.v_ = None

        self.err_y.initialize(self.device)
        self.n_err_skipped.initialize(self.device)
        self.target.initialize(self.device)
        self.metrics.initialize(self.device)
        self.mse.initialize(self.device)

        if not self.device:
            return

        self.krn_constants_d_ = numpy.zeros(2, config.dtypes[config.dtype])
        self.krn_constants_i_ = numpy.zeros(1, config.itypes[itype2])

        if self.prg_ == None:
            defines = ("%s\n"
                       "#define BLOCK_SIZE %d\n"
                       "#define BATCH %d\n"
                       "#define Y %d\n"
                       "#define Y_REAL %d\n\n") % \
                   (config.cl_defines[config.dtype],
                    self.device.info.BLOCK_SIZE[config.dtype],
                    self.err_y.aligned_.shape[0],
                    self.err_y.aligned_.size // self.err_y.aligned_.shape[0],
                    self.err_y.v.size // self.err_y.v.shape[0])
            s = defines
            for src, define in self.cl_sources_.items():
                s += "\n" + define + "\n"
                fin = open(src, "r")
                s += fin.read()
                fin.close()
            fout = open("%s/ev_%d.cl" % (config.cache_dir,
                self.y.v.size // self.y.v.shape[0]), "w")
            fout.write(s)
            fout.close()

            self.prg_ = pyopencl.Program(self.device.context_, s).build()

            self.krn_ = pyopencl.Kernel(self.prg_, "ev_mse")
            self.krn_.set_arg(0, self.y.v_)
            self.krn_.set_arg(1, self.target.v_)
            self.krn_.set_arg(2, self.err_y.v_)
            self.krn_.set_arg(3, self.n_err_skipped.v_)
            self.krn_.set_arg(4, self.metrics.v_)
            self.krn_.set_arg(8, self.mse.v_)

    def gpu_run(self):
        # return self.cpu_run()
        t1 = time.time()

        batch_size = self.batch_size[0]

        if self.class_target != None:
            self.n_err_skipped.sync()
            old_n_err = self.n_err_skipped.v[0]

        self.y.sync(formats.GPU)
        self.target.sync(formats.GPU)
        self.n_err_skipped.sync(formats.GPU)
        self.metrics.sync(formats.GPU)

        self.krn_constants_i_[0] = batch_size
        self.krn_constants_d_[0] = self.threshold_skip
        self.krn_constants_d_[1] = self.threshold_ok
        self.krn_.set_arg(5, self.krn_constants_i_[0])
        self.krn_.set_arg(6, self.krn_constants_d_[0])
        self.krn_.set_arg(7, self.krn_constants_d_[1])

        local_size = [self.device.info.BLOCK_SIZE[config.dtype]]
        global_size = [local_size[0]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                                 self.krn_,
                                                 global_size, local_size)
        event.wait()

        self.err_y.update(formats.GPU)
        self.n_err_skipped.update(formats.GPU)
        # self.n_err_skipped.sync()
        # self.effective_batch_size[0] = (self.batch_size[0] -
        #    self.n_err_skipped.v[1])
        self.effective_batch_size[0] = self.batch_size[0]
        self.metrics.update(formats.GPU)
        self.mse.update(formats.GPU)

        if self.class_target != None:
            n_err = 0
            self.y.sync()
            self.n_err_skipped.sync()
            t = self.class_target.v
            y = self.y.v
            for i in range(0, y.shape[0]):
                md = 1.0e30
                mi = 0
                for j in range(0, t.shape[0]):
                    d = numpy.linalg.norm(t[j] - y[i])
                    if d < md:
                        md = d
                        mi = j
                if mi != self.labels.v[i]:
                    n_err += 1
            self.n_err_skipped.v[0] = old_n_err + n_err
            self.n_err_skipped.update()

        self.log().debug("%s in %.2f sec" % (self.__class__.__name__,
                                             time.time() - t1))

    def cpu_run(self):
        return self.gpu_run()
