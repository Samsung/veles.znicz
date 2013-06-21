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

    Creates within initialize():
        err_y
        n_err_skipped
        confusion_matrix
        skipped

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
        krn_constants_d_: numpy array for constant arguments to kernel.
        krn_constants_i_: numpy array for constant arguments to kernel.
    """
    def __init__(self, threshold=1.0, threshold_low=None, device=None,
                 compute_confusion_matrix=True, unpickling=0):
        super(EvaluatorSoftmax, self).__init__(unpickling=unpickling,
                                               device=device)
        if unpickling:
            return
        self.labels = None  # formats.Labels()
        self.y = None  # formats.Batch()
        self.err_y = formats.Batch()
        self.batch_size = None  # [0]
        self.max_samples_per_epoch = None  # [0]
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.skipped = formats.Batch()
        self.n_skipped = None
        self.compute_confusion_matrix = compute_confusion_matrix
        self.confusion_matrix = formats.Vector()
        self.n_err_skipped = formats.Vector()
        self.max_idx = None  # formats.Batch()
        self.krn_constants_d_ = None
        self.krn_constants_i_ = None

    def initialize(self):
        itype = config.get_itype_from_size((self.y.batch.size //
                                            self.y.batch.shape[0]))
        itype2 = config.get_itype_from_size(self.max_samples_per_epoch[0])
        self.cl_sources["cl/ev.cl"] = ("#define itype %s\n#define itype2 %s" %
            (itype, itype2))

        if (self.err_y.batch == None or
            self.err_y.batch.size != self.y.batch.size):
            self.err_y.batch = numpy.zeros(self.y.batch.shape,
                dtype=config.dtypes[config.dtype])
            self.err_y.batch_ = None

        if self.n_err_skipped.v == None or self.n_err_skipped.v.size < 2:
            self.n_err_skipped.v = numpy.zeros(2, dtype=config.itypes[itype2])
            self.n_err_skipped.v_ = None

        if (self.skipped.batch == None or
            self.skipped.batch.size != self.y.batch.shape[0]):
            self.skipped.batch = numpy.zeros(self.y.batch.shape[0],
                dtype=numpy.int8)
            self.skipped.batch_ = None

        out_size = self.y.batch.size // self.y.batch.shape[0]
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

        self.err_y.initialize(self.device)
        self.confusion_matrix.initialize(self.device)
        self.skipped.initialize(self.device)
        self.n_err_skipped.initialize(self.device)
        self.max_idx.initialize(self.device)
        self.labels.initialize(self.device)

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
                    self.err_y.batch.size // self.err_y.batch.shape[0])
            s = defines
            for src, define in self.cl_sources.items():
                if type(define) == type(""):
                    s += "\n" + define + "\n"
                fin = open(src, "r")
                s += fin.read()
                fin.close()
            fout = open("cache/ev_%d.cl" % (self.y.batch.size //
                                            self.y.batch.shape[0], ), "w")
            fout.write(s)
            fout.close()

            self.prg_ = pyopencl.Program(self.device.context_, s).build()

            self.krn_ = pyopencl.Kernel(self.prg_, "ev_sm")
            self.krn_.set_arg(0, self.y.batch_)
            self.krn_.set_arg(1, self.max_idx.batch_)
            self.krn_.set_arg(2, self.labels.batch_)
            self.krn_.set_arg(3, self.err_y.batch_)
            self.krn_.set_arg(4, self.skipped.batch_)
            self.krn_.set_arg(5, self.n_err_skipped.v_)
            self.krn_.set_arg(6, self.confusion_matrix.v_)

    def gpu_run(self):
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

        self.krn_constants_i_[0] = batch_size
        self.krn_constants_d_[0] = threshold
        self.krn_constants_d_[1] = threshold_low
        self.krn_.set_arg(7, self.krn_constants_i_[0])
        self.krn_.set_arg(8, self.krn_constants_d_[0])
        self.krn_.set_arg(9, self.krn_constants_d_[1])

        local_size = [self.device.info.BLOCK_SIZE[config.dtype]]
        global_size = [local_size[0]]
        event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                                                 self.krn_,
                                                 global_size, local_size)
        event.wait()

        self.err_y.update(formats.GPU)
        self.confusion_matrix.update(formats.GPU)
        self.skipped.update(formats.GPU)
        self.n_err_skipped.update(formats.GPU)

        if __debug__:
            print("%s in %.2f sec" % (self.__class__.__name__,
                                      time.time() - t1))

    def cpu_run(self):
        t1 = time.time()
        self.y.sync()
        self.max_idx.sync()
        batch_size = self.batch_size[0]
        labels = self.labels.batch

        threshold = self.threshold
        threshold_low = self.threshold_low
        if threshold_low == None:
            threshold_low = threshold

        confusion_matrix = self.confusion_matrix.v
        self.confusion_matrix.sync()
        self.skipped.sync()

        n_ok = 0
        n_skip = 0
        for i in range(0, batch_size):  # loop by batch
            y = self.y.batch[i]
            y = y.reshape(y.size)  # make it plain
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain

            skip = False
            max_idx = self.max_idx.batch[i]
            max_vle = y[max_idx]
            confusion_matrix[max_idx, labels[i]] += 1
            if max_idx == labels[i]:
                n_ok += 1
                # check for threshold
                if (max_vle > threshold) or \
                   ((max_vle > threshold_low) and (self.skipped[i])):
                    err_y[:] = 0  # already trained good enough, skip it
                    self.skipped.batch[i] = 1
                    skip = True
                    n_skip += 1

            if not skip:
                # Compute softmax output error gradient
                err_y[:] = y[:]
                err_y[labels[i]] = y[labels[i]] - 1.0
                self.skipped.batch[i] = 0
        # Set errors for excessive samples to zero
        if batch_size < self.err_y.batch.shape[0]:
            err_y = self.err_y.batch[batch_size:]
            err_y = err_y.reshape(err_y.size)  # make it plain
            err_y[:] = 0.0
        self.n_err_skipped.v[0] += batch_size - n_ok
        self.n_err_skipped.v[1] += n_skip

        self.err_y.update()
        self.confusion_matrix.update()
        self.n_err_skipped.update()
        if __debug__:
            print("%s in %.2f sec" % (self.__class__.__name__,
                                      time.time() - t1))


class EvaluatorMSE(units.OpenCLUnit):
    """MSE Evaluator for nn output from the batch labels.

    TODO(a.kazantsev): make it proper.

    Attributes:
        y_original: actual function values for a batch.
        labels: Labels as in softmax (may be defined instead of "a",
                in such case "a" will be created in initialize()).
        y: output of the network as Batch.
        err_y: backpropagation errors based on labels.
        status: status of the evaluation
                (status.completed = True when learning ended).
        mse_stop: target mse for all samples within a batch.
    """
    def __init__(self, device=None, mse_stop=0.3, unpickling=0):
        super(EvaluatorMSE, self).__init__(unpickling=unpickling,
                                           device=device)
        self.save_failed = False
        self.first_run = True
        if unpickling:
            return
        self.y_original = None  # formats.Batch(device)
        self.labels = None  # formats.Labels()
        self.y = None  # formats.Batch(device)
        self.err_y = formats.Batch()
        self.status = units.Connector()
        self.status.completed = False
        self.status.n_ok = 0
        self.status.num_errors = 0
        self.mse_stop = mse_stop

    def initialize(self):
        if self.err_y.batch == None or \
           self.err_y.batch.size != self.y.batch.size:
            self.err_y.batch = numpy.zeros(self.y.batch.shape,
                                           dtype=config.dtypes[config.dtype])
            self.err_y.batch_ = None
        self.err_y.initialize(self.device)

        if self.y_original == None and self.labels != None:
            self.y_original = formats.Batch()
            self.y_original.batch = numpy.zeros(self.y.batch.shape,
                dtype=config.dtypes[config.dtype])
            batch_size = self.y.batch.shape[0]
            y_original = self.y_original.batch
            y_original = y_original.reshape([batch_size,
                                             y_original.size // batch_size])
            labels = self.labels.batch
            for i in range(0, batch_size):
                sample = y_original[i]
                sample[:] = -1.6
                sample[labels[i]] = 1.6
        self.y_original.initialize(self.device)

    def cpu_run(self):
        t1 = time.time()

        self.y.sync()
        n_ok = 0
        n_skip = 0
        batch_size = self.y.batch.shape[0]
        y = self.y.batch
        y = y.reshape([batch_size, y.size // batch_size])
        y_original = self.y_original.batch
        y_original = y_original.reshape([batch_size,
                                         y_original.size // batch_size])
        err_y = self.err_y.batch
        err_y = err_y.reshape([batch_size, err_y.size // batch_size])

        numpy.subtract(y, y_original, err_y)

        max_mse = 0
        maxdiff = numpy.max(numpy.abs(err_y))
        for i in range(0, batch_size):
            sample = err_y[i]
            mse = numpy.linalg.norm(sample) / sample.size
            if mse > max_mse:
                max_mse = mse
            diff = numpy.max(numpy.abs(sample))
            if diff < self.mse_stop:
                n_ok += 1

        self.err_y.update()
        self.status.n_ok = n_ok
        self.status.completed = False
        self.status.num_errors = batch_size - n_ok  # Number of errors
        self.status.update()
        print("(n_ok, n_total, max_mse, maxdiff): (%d, %d, %.6f, %.6f)" %
              (n_ok, batch_size, max_mse, maxdiff))
        if not self.first_run and (True or self.threshold == 1.0 or \
                                   n_skip == batch_size) and \
           n_ok == batch_size:
            print("Perfect")
            self.status.completed = True
            self.status.update()
            return
        self.first_run = False

        dt = time.time() - t1
        if not __debug__:
            print("Computed softmax errs within %.2f sec, skipped %.2f%%" %
                  (dt, n_skip / batch_size * 100.0))
            return
        err_y = self.err_y.batch
        print("Computed softmax errs within %.2f sec, skipped %.2f%%: "
              "(min, max, avg) = (%.3f, %.3f, %.3f)" %
              (dt, n_skip / batch_size * 100.0, err_y.min(), err_y.max(),
               numpy.average(err_y)))


class EvaluatorSoftmax2(units.OpenCLUnit):
    """Evaluator for nn softmax output from the batch labels.

    Attributes:
        labels: labels for Batch.
        y: output of the network as Batch.
        err_y: backpropagation errors based on labels.
        status: status of the evaluation
                (status.completed = True when learning ended).
        threshold: when output becomes greater than this value,
                   assume gradient as 0.
        threshold_low: when gradient was assumed as 0 and output becomes less
                       than this value, calculate gradient as usual.
        skipped: array of bytes with non-zero value if the sample was skipped
                 due to assumed zero-gradient.
    """
    def __init__(self, threshold=0.33, threshold_low=None, device=None,
                 unpickling=0):
        super(EvaluatorSoftmax2, self).__init__(unpickling=unpickling,
                                               device=device)
        self.save_failed = False
        self.first_run = True
        if unpickling:
            return
        self.labels = None  # formats.Labels()
        self.y = None  # formats.Batch(device)
        self.err_y = formats.Batch()
        self.status = units.Connector()
        self.status.completed = False
        self.status.n_ok = 0
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.skipped = None

        self.params = None
        self.TrainIndex = None
        self.ValidIndex = None
        self.TestIndex = None
        self.err_y_o = formats.Batch()
        self.err_y_v = formats.Batch()
        self.err_y_t = formats.Batch()
        self.L = units.Connector()
        self.L.value = 0
        #self.Index=None

    def initialize(self):
        if self.err_y.batch == None or \
           self.err_y.batch.size != self.y.batch.size:
            self.err_y.batch = numpy.zeros(self.y.batch.shape,
                                           dtype=config.dtypes[config.dtype])
            self.err_y.batch_ = None
        self.skipped = numpy.zeros([self.y.batch.shape[0]], dtype=numpy.byte)
        _t = self.params['data_set']

        self._ff = _t['type1']
        if self._ff in [1, 2]:
            self.use_valid = 1
        else:
            self.use_valid = 0
        if self._ff in [2, 3]:
            self.use_test = 1
        else:
            self.use_test = 0

        self.err_y.initialize(self.device)

        if self.use_valid == 1:
            self.err_y_v.batch = numpy.zeros(self.y.batch.shape,
                                             dtype=config.dtypes[config.dtype])
            self.count_valid = numpy.sum(self.ValidIndex.batch)
        if self.use_test == 1:
            self.err_y_t.batch = numpy.zeros(self.y.batch.shape,
                                             dtype=config.dtypes[config.dtype])
            self.count_test = numpy.sum(self.TestIndex.batch)

        self.err_y_o.batch = numpy.zeros(self.y.batch.shape,
                                         dtype=config.dtypes[config.dtype])
        self.count_train = numpy.sum(self.TrainIndex.batch)
        self.L.value = self.count_train

    def cpu_run(self):
        t1 = time.time()

        self.y.sync()
        n_ok = 0
        n_ok_v = 0
        n_ok_t = 0
        n_skip = 0
        batch_size = self.y.batch.shape[0]
        labels = self.labels.batch

        threshold = self.threshold
        threshold_low = self.threshold_low
        if threshold_low == None:
            threshold_low = threshold
        #Index = self.Index
        TrainIndex = self.TrainIndex
        TestIndex = self.TestIndex
        ValidIndex = self.ValidIndex

        print(" self.count_train = ", self.count_train,
              " ", self.use_valid, " ", self.use_test)
        for i in range(0, batch_size):  # loop by batch
            y = self.y.batch[i]
            y = y.reshape(y.size)  # make it plain
            err_y = self.err_y.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            if TrainIndex.batch[i] == 1:
                skip = False
                i_max = numpy.argmax(y)

                if i_max == labels[i]:
                    n_ok += 1
                    # check for threshold
                    if (y[i_max] >= threshold) or \
                       ((y[i_max] >= threshold_low) and (self.skipped[i])):
                        err_y[:] = 0  # already trained good enough, skip it
                        self.skipped[i] = 1
                        skip = True
                        n_skip += 1

                if not skip:
                    # Compute softmax output error gradient
                    err_y[:] = y[:]
                    err_y[labels[i]] = y[labels[i]] - 1.0
                    self.skipped[i] = 0
            else:
                err_y[:] = 0

        self.err_y.update()
        #print(self.err_y.batch)

        if self.use_valid == 1:
            for i in range(0, batch_size):  # loop by batch
                y = self.y.batch[i]
                y = y.reshape(y.size)  # make it plain
                err_y = self.err_y_v.batch[i]
                err_y = err_y.reshape(err_y.size)  # make it plain
                if ValidIndex.batch[i] == 1:
                    i_max = numpy.argmax(y)
                    if i_max == labels[i]:
                        n_ok_v += 1
                    # Compute softmax output error gradient
                    err_y[:] = y[:]
                    err_y[labels[i]] = y[labels[i]] - 1.0
                else:
                    err_y[:] = 0

        if self.use_test == 1:
            for i in range(0, batch_size):  # loop by batch
                y = self.y.batch[i]
                y = y.reshape(y.size)  # make it plain
                err_y = self.err_y_t.batch[i]
                err_y = err_y.reshape(err_y.size)  # make it plain
                if TestIndex.batch[i] == 1:
                    i_max = numpy.argmax(y)
                    if i_max == labels[i]:
                        n_ok_t += 1
                    # Compute softmax output error gradient
                    err_y[:] = y[:]
                    err_y[labels[i]] = y[labels[i]] - 1.0
                else:
                    err_y[:] = 0
        for i in range(0, batch_size):  # loop by batch
            y = self.y.batch[i]
            y = y.reshape(y.size)  # make it plain
            err_y = self.err_y_o.batch[i]
            err_y = err_y.reshape(err_y.size)  # make it plain
            if TrainIndex.batch[i] == 1:
                i_max = numpy.argmax(y)
                # Compute softmax output error gradient
                err_y[:] = y[:]
                err_y[labels[i]] = y[labels[i]] - 1.0
            else:
                err_y[:] = 0

        err_y = self.err_y_o.batch
        print("Computed softmax for train errs (min, max, avg, mean, std) = "
              "(%.3f, %.3f, %.3f , %.3f , %.3f)" % \
              (err_y.min(), err_y.max(), numpy.average(err_y),
               numpy.mean(err_y), numpy.std(err_y)))
        if self.use_valid == 1:
            err_y = self.err_y_v.batch
            print("Computed softmax for valid errs (min, max, avg, mean, std)"
                  " = (%.3f, %.3f, %.3f , %.3f , %.3f)" % \
                  (err_y.min(), err_y.max(), numpy.average(err_y),
                   numpy.mean(err_y), numpy.std(err_y)))
        if self.use_test == 1:
            err_y = self.err_y_t.batch
            print("Computed softmax for test errs (min, max, avg, mean, std)"
                  " = (%.3f, %.3f, %.3f , %.3f , %.3f)" % \
                  (err_y.min(), err_y.max(), numpy.average(err_y),
                   numpy.mean(err_y), numpy.std(err_y)))

        self.status.n_ok = n_ok
        self.status.count_train = self.count_train
        if self.use_valid == 1:
            self.status.n_ok_v = n_ok_v
            self.status.count_valid = self.count_valid
        if self.use_test == 1:
            self.status.n_ok_t = n_ok_t
            self.status.count_test = self.count_test
        self.status.completed = False

        if self.use_valid == 0 and self.use_test == 0:
            print("RESULT: (%d:%d | %d)" % \
                  (n_ok, self.count_train, batch_size))
        if self.use_valid == 1 and self.use_test == 0:
            self.count_valid
            print("RESULT: (%d:%d | %d:%d | %d)" % \
                  (n_ok, self.count_train, n_ok_v, self.count_valid,
                   batch_size))
        if self.use_test == 1 and self.use_valid == 0:
            print("RESULT: (%d:%d | %d:%d | %d)" % \
                  (n_ok, self.count_train, n_ok_t, self.count_test,
                   batch_size))

        if self.use_test == 1 and self.use_valid == 1:
            print("RESULT: (%d:%d | %d:%d | %d :%d | %d)" % \
                  (n_ok, self.count_train, n_ok_v, self.count_valid,
                   n_ok_t, self.count_test, batch_size))

        if not self.first_run and \
           (True or self.threshold == 1.0 or n_skip == self.count_train) and \
           n_ok == self.count_train:
            print("Perfect")
            self.status.completed = True
            self.status.update()
            return
        self.first_run = False

        dt = time.time() - t1
        if not __debug__:
            print("Computed softmax errs within %.2f sec, skipped %.2f%%" % \
                  (dt, n_skip / batch_size * 100.0))
            return

        err_y = self.err_y.batch
        print("Computed softmax errs within %.2f sec, skipped %.2f%%: "
              "(min, max, avg, mean, std) = "
              "(%.3f, %.3f, %.3f , %.3f , %.3f)" % \
              (dt, n_skip / batch_size * 100.0, err_y.min(), err_y.max(),
               numpy.average(err_y), numpy.mean(err_y), numpy.std(err_y)))
