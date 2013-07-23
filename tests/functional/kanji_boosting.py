#!/usr/bin/python3.3 -O
"""
Created on June 29, 2013

File for kanji recognition.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import os
import struct
import logging


class BMPWriter(object):
    """Writes bmp file.

    Attributes:
        rgbquad: color table for gray scale.
    """
    def __init__(self):
        self.rgbquad = numpy.zeros([256, 4], dtype=numpy.uint8)
        for i in range(0, 256):
            self.rgbquad[i] = (i, i, i, 0)

    def write_gray(self, fnme, a):
        """Writes bmp as gray scale.

        Parameters:
            fnme: file name.
            a: numpy array with 2 dimensions.
        """
        if len(a.shape) != 2:
            raise Exception("a should be 2-dimensional, got: %s" % (
                str(a.shape),))

        fout = open(fnme, "wb")

        header_size = 54 + 256 * 4
        file_size = header_size + a.size

        header = struct.pack("<HIHHI"
                             "IiiHHIIiiII",
                             19778, file_size, 0, 0, header_size,
                             40, a.shape[1], -a.shape[0], 1, 8, 0, a.size,
                             0, 0, 0, 0)
        fout.write(header)

        self.rgbquad.tofile(fout)

        a.astype(numpy.uint8).tofile(fout)

        fout.close()


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s/../src" % (this_dir,))
add_path("%s/../.." % (this_dir,))
add_path("%s/../../../src" % (this_dir,))


import units
import formats
import error
import numpy
import config
import rnd
import opencl
import plotters
import pickle
import time
import glob
#import scipy.ndimage


def normalize(a):
    a -= a.min()
    m = a.max()
    if m:
        a /= m
        a *= 2.0
        a -= 1.0


class Loader(units.Unit):
    """Loads Hands data and provides mini-batch output interface.

    Attributes:
        rnd: rnd.Rand().

        minibatch_data: Hands images scaled to [-1, 1].
        minibatch_indexes: global indexes of images in minibatch.
        minibatch_labels: labels for indexes in minibatch.

        minibatch_class: class of the minibatch: 0-test, 1-validation, 2-train.
        minibatch_last: if current minibatch is last in it's class.

        minibatch_offs: offset of the current minibatch in all samples,
                        where first come test samples, then validation, with
                        train ones at the end.
        minibatch_size: size of the current minibatch.
        total_samples: total number of samples in the dataset.
        class_samples: number of samples per class.
        minibatch_maxsize: maximum size of minibatch in samples.
        nextclass_offs: offset in samples where the next class begins.

        original_data: original Hands images scaled to [-1, 1] as single batch.
        original_labels: original Hands labels as single batch.
        width: width of the input image.
    """
    def __init__(self,
                 train_paths=["%s/kanji/train/*.bmp" % (
                    config.test_dataset_root,)],
                 target_dir=("%s/kanji/target" % (config.test_dataset_root,)),
                 minibatch_max_size=100, rnd=rnd.default, unpickling=0):
        super(Loader, self).__init__(unpickling=unpickling)
        if unpickling:
            self.target_by_lbl = {}
            return
        self.width = None

        self.train_paths = train_paths
        self.target_dir = target_dir

        self.rnd = [rnd]

        self.minibatch_data = formats.Batch()
        self.minibatch_target = formats.Batch()
        self.minibatch_indexes = formats.Labels()
        self.minibatch_labels = formats.Labels(1000)

        self.minibatch_class = [0]
        self.minibatch_last = [0]

        self.total_samples = [0]
        self.class_samples = [0, 0, 0]
        self.minibatch_offs = [0]
        self.minibatch_size = [0]
        self.minibatch_maxsize = [minibatch_max_size]
        self.nextclass_offs = [0, 0, 0]

        self.original_data = None
        self.original_labels = None
        self.original_target = None
        self.target_by_lbl = {}

        self.shuffled_indexes = None

    def __getstate__(self):
        state = super(Loader, self).__getstate__()
        state["original_data"] = None
        state["original_labels"] = None
        state["original_target"] = None
        state["shuffled_indexes"] = None
        return state

    def from_image(self, fnme):
        #a = scipy.ndimage.imread(fnme)
        fin = open(fnme, "rb")
        fin.read(54 + 256 * 4)
        a = numpy.fromfile(fin, dtype=numpy.uint8)
        sx = int(numpy.round(numpy.sqrt(a.size)))
        sy = int(numpy.round(a.size / sx))
        a = a.reshape(sy, sx)
        fin.close()

        a = a.astype(config.dtypes[config.dtype])
        normalize(a)
        #a /= 127.5
        #a -= 1.0
        #if numpy.fabs(a.min() + 1.0) > 0.000001 or \
        #   numpy.fabs(a.max() - 1.0) > 0.000001:
        #    self.log().info("Found an image with unexpected range:",
        #        a.min(), a.max(), fnme)
        return a

    def load_original(self, pathname):
        """Loads data from original files.
        """
        self.log().info("Loading from %s..." % (pathname,))
        files = glob.glob(pathname)
        files.sort()
        n_files = len(files)
        if not n_files:
            raise error.ErrNotExists("No files fetched as %s" % (pathname,))

        a = self.from_image(files[0])
        sh = [n_files]
        for x in a.shape:
            sh.append(x)
        aa = numpy.zeros(sh, dtype=config.dtypes[config.dtype])
        ll = numpy.zeros(n_files, dtype=numpy.int16)
        aa[0][:] = a[:]
        st = len(pathname) - 5
        s = files[0][st:]
        zi = 0
        while s[zi] == '0':
            zi += 1
        lbl = int(s[zi:5]) - 1
        ll[0] = lbl
        if lbl not in self.target_by_lbl:
            b = self.from_image("%s/%s.bmp" % (self.target_dir, s[:5]))
            self.target_by_lbl[lbl] = b
        for i in range(1, n_files):
            b = self.from_image(files[i])
            if b.size != a.size:
                raise error.ErrBadFormat("Found file with different "
                                         "size than first: %s", files[i])
            aa[i][:] = b[:]
            s = files[i][st:]
            zi = 0
            while s[zi] == '0':
                zi += 1
            lbl = int(s[zi:5]) - 1
            ll[i] = lbl
            if lbl not in self.target_by_lbl:
                b = self.from_image("%s/%s.bmp" % (self.target_dir, s[:5]))
                self.target_by_lbl[lbl] = b
        self.log().info("Labels are indexed from-to: %d %d" % (ll.min(),
                                                               ll.max()))
        return (aa, ll)

    def initialize(self):
        """Here we will load Hands data.
        """
        #if self.original_data != None and self.original_labels != None:
        #    return
        # Load validation set.
        self.class_samples[0] = 0
        self.nextclass_offs[0] = 0
        self.class_samples[1] = 0
        self.nextclass_offs[1] = 0
        # Load train set.
        a, l = self.load_original(self.train_paths[0])
        self.original_data = a
        self.original_labels = l
        for i in range(1, len(self.train_paths)):
            a, l = self.load_original(self.train_paths[i])
            self.original_data = numpy.append(self.original_data, a, 0)
            self.original_labels = numpy.append(self.original_labels, l, 0)
        self.total_samples[0] = self.original_data.shape[0]
        self.class_samples[2] = self.total_samples[0] - self.class_samples[1]
        self.nextclass_offs[2] = self.total_samples[0]

        sh = [len(self.original_labels)]
        for i in self.target_by_lbl[self.original_labels[0]].shape:
            sh.append(i)
        self.original_target = numpy.zeros(sh,
                                    dtype=config.dtypes[config.dtype])
        for i in range(0, len(self.original_labels)):
            self.original_target[i][:] = \
                self.target_by_lbl[self.original_labels[i]][:]
        self.target_by_lbl = {}

        self.shuffled_indexes = numpy.arange(self.total_samples[0],
                                             dtype=numpy.int32)

        self.minibatch_maxsize[0] = min(self.minibatch_maxsize[0],
                                        max(self.class_samples[2],
                                            self.class_samples[1],
                                            self.class_samples[0]))

        sh = [self.minibatch_maxsize[0]]
        for i in self.original_data.shape[1:]:
            sh.append(i)
        self.minibatch_data.batch = numpy.zeros(
            sh, dtype=config.dtypes[config.dtype])
        sht = [self.minibatch_maxsize[0]]
        for i in self.original_target.shape[1:]:
            sht.append(i)
        self.minibatch_target.batch = numpy.zeros(
            sht, dtype=config.dtypes[config.dtype])
        self.minibatch_labels.batch = numpy.zeros(
            [self.minibatch_maxsize[0]], dtype=numpy.int8)
        self.minibatch_indexes.batch = numpy.zeros(
            [self.minibatch_maxsize[0]], dtype=numpy.int32)

        self.minibatch_indexes.n_classes = self.total_samples[0]
        self.minibatch_labels.n_classes = self.original_labels.max() + 1

        self.minibatch_offs[0] = self.total_samples[0]

        if self.class_samples[0]:
            self.shuffle_validation_train()
        else:
            self.shuffle_train()

    def shuffle_validation_train(self):
        self.rnd[0].shuffle(self.shuffled_indexes[self.nextclass_offs[0]:\
                                                  self.nextclass_offs[2]])

    def shuffle_train(self):
        self.rnd[0].shuffle(self.shuffled_indexes[self.nextclass_offs[1]:\
                                                  self.nextclass_offs[2]])

    def shuffle(self):
        """Shuffle the dataset after one epoch.
        """
        self.shuffle_train()

    def run(self):
        """Prepare the minibatch.
        """
        t1 = time.time()

        self.minibatch_offs[0] += self.minibatch_size[0]
        # Reshuffle when end of data reached.
        if self.minibatch_offs[0] >= self.total_samples[0]:
            self.shuffle()
            self.minibatch_offs[0] = 0

        # Compute minibatch size and it's class.
        for i in range(0, len(self.nextclass_offs)):
            if self.minibatch_offs[0] < self.nextclass_offs[i]:
                self.minibatch_class[0] = i
                minibatch_size = min(self.minibatch_maxsize[0],
                    self.nextclass_offs[i] - self.minibatch_offs[0])
                if self.minibatch_offs[0] + minibatch_size >= \
                   self.nextclass_offs[self.minibatch_class[0]]:
                    self.minibatch_last[0] = 1
                else:
                    self.minibatch_last[0] = 0
                break
        else:
            raise error.ErrNotExists("Could not determine minibatch class.")
        self.minibatch_size[0] = minibatch_size

        # Fill minibatch data labels and indexes according to current shuffle.
        idxs = self.minibatch_indexes.batch
        idxs[0:minibatch_size] = self.shuffled_indexes[self.minibatch_offs[0]:\
            self.minibatch_offs[0] + minibatch_size]

        self.minibatch_labels.batch[0:minibatch_size] = \
            self.original_labels[idxs[0:minibatch_size]]

        self.minibatch_data.batch[0:minibatch_size] = \
            self.original_data[idxs[0:minibatch_size]]

        self.minibatch_target.batch[0:minibatch_size] = \
            self.original_target[idxs[0:minibatch_size]]

        # Fill excessive indexes.
        if minibatch_size < self.minibatch_maxsize[0]:
            self.minibatch_data.batch[minibatch_size:] = 0.0
            self.minibatch_target.batch[minibatch_size:] = 0.0
            self.minibatch_labels.batch[minibatch_size:] = -1
            self.minibatch_indexes.batch[minibatch_size:] = -1

        # Set update flag for GPU operation.
        self.minibatch_data.update()
        self.minibatch_target.update()
        self.minibatch_labels.update()
        self.minibatch_indexes.update()

        self.log().debug("%s in %.2f sec" % (self.__class__.__name__,
                                      time.time() - t1))


import all2all
import evaluator
import gd


class ImageSaverAE(units.Unit):
    """Saves input and output side by side to png for AutoEncoder.

    Attributes:
        out_dirs: output directories by minibatch_class where to save png.
        input: batch with input samples.
        output: batch with corresponding output samples.
        indexes: sample indexes.
        labels: sample labels.
    """
    def __init__(self, out_dirs=["/tmp/img/test", "/tmp/img/validation",
                                 "/tmp/img/train"], unpickling=0):
        super(ImageSaverAE, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.out_dirs = out_dirs
        self.input = None  # formats.Batch()
        self.output = None  # formats.Batch()
        self.indexes = None  # formats.Batch()
        self.labels = None  # formats.Batch()
        self.minibatch_class = None  # [0]
        self.minibatch_size = None  # [0]
        self.last_save_time = 0
        self.this_save_time = [0]
        self.bmp = BMPWriter()

    def initialize(self):
        for dirnme in self.out_dirs:
            try:
                os.mkdir(dirnme)
            except OSError:
                pass

    def run(self):
        self.input.sync()
        self.output.sync()
        self.indexes.sync()
        self.labels.sync()
        xy = None
        dirnme = self.out_dirs[self.minibatch_class[0]]
        if self.this_save_time[0] > self.last_save_time:
            self.last_save_time = self.this_save_time[0]
            files = glob.glob("%s/*.bmp" % (dirnme,))
            for file in files:
                try:
                    os.unlink(file)
                except FileNotFoundError:
                    pass
        for i in range(0, self.minibatch_size[0]):
            x = self.input.batch[i]
            y = self.output.batch[i].reshape(x.shape)
            idx = self.indexes.batch[i]
            lbl = self.labels.batch[i]
            d = x - y
            mse = numpy.linalg.norm(d) / d.size
            if xy == None:
                xy = numpy.empty([x.shape[0] * 2, x.shape[1] * 3],
                                 dtype=x.dtype)
            xy[:x.shape[0], :x.shape[1]] = x[:, :]
            xy[:x.shape[0], x.shape[1]:x.shape[1] * 2] = y[:, :]
            xy[:x.shape[0], x.shape[1] * 2:] = d[:, :]
            x2 = xy[:x.shape[0], :x.shape[1]]
            y2 = xy[:x.shape[0], x.shape[1]:x.shape[1] * 2]
            d2 = xy[:x.shape[0], x.shape[1] * 2:]
            #
            xy[x.shape[0]:, :x.shape[1]] = x[:, :]
            xy[x.shape[0]:, x.shape[1]:x.shape[1] * 2] = y[:, :]
            xy[x.shape[0]:, x.shape[1] * 2:] = d[:, :]
            x21 = xy[x.shape[0]:, :x.shape[1]]
            y21 = xy[x.shape[0]:, x.shape[1]:x.shape[1] * 2]
            d21 = xy[x.shape[0]:, x.shape[1] * 2:]
            #x *= -1.0
            x2 += 1.0
            x2 *= 127.5
            numpy.clip(x2, 0, 255, x2)
            #y *= -1.0
            y2 += 1.0
            y2 *= 127.5
            numpy.clip(y2, 0, 255, y2)
            #
            normalize(d2)
            d2 += 1.0
            d2 *= 127.5
            #
            normalize(x21)
            x21 += 1.0
            x21 *= 127.5
            #
            normalize(y21)
            y21 += 1.0
            y21 *= 127.5
            #
            normalize(d21)
            d21 += 1.0
            d21 *= 127.5
            #fnme = "%s/%.6f_%d_%d.png" % (
            #    self.out_dirs[self.minibatch_class[0]], mse, lbl, idx)
            #scipy.misc.imsave(fnme, xy.astype(numpy.uint8))
            #fnme = "%s/out_%d.png" % (self.out_dirs[self.minibatch_class[0]],
            #                          idx)
            #scipy.misc.imsave(fnme, y2.astype(numpy.uint8))
            fnme = "%s/%06f.%05d.%06d.bmp" % (dirnme, mse, lbl, idx)
            tmp = y2.astype(numpy.uint8).reshape(x.shape)
            self.bmp.write_gray(fnme, tmp)


class Decision(units.Unit):
    """Decides on the learning behavior.

    Attributes:
        complete: completed.
        minibatch_class: current minibatch class.
        minibatch_last: if current minibatch is last in it's class.
        gd_skip: skip gradient descent or not.
        epoch_number: epoch number.
        epoch_min_mse: minimum sse by class per epoch.
        minibatch_n_err: number of errors for minibatch.
        minibatch_metrics: [0] - sse, [1] - max of sum of sample graidents.
        class_samples: number of samples per class.
        epoch_ended: if an epoch has ended.
        fail_iterations: number of consequent iterations with non-decreased
            validation error.
        epoch_metrics: metrics for each set epoch.
    """
    def __init__(self, fail_iterations=10000, unpickling=0):
        super(Decision, self).__init__(unpickling=unpickling)
        if unpickling:
            self.epoch_min_mse = [1.0e30, 1.0e30, 1.0e30]
            self.n_err = [1.0e30, 1.0e30, 1.0e30]
            self.n_err_pt = [100.0, 100.0, 100.0]
            return
        self.complete = [0]
        self.minibatch_class = None  # [0]
        self.minibatch_last = None  # [0]
        self.gd_skip = [0]
        self.epoch_number = [0]
        self.epoch_min_mse = [1.0e30, 1.0e30, 1.0e30]
        self.n_err = [1.0e30, 1.0e30, 1.0e30]
        self.minibatch_n_err = None  # formats.Vector()
        self.minibatch_metrics = None  # formats.Vector()
        self.fail_iterations = [fail_iterations]
        self.epoch_ended = [0]
        self.n_err_pt = [100.0, 100.0, 100.0]
        self.class_samples = None  # [0, 0, 0]
        self.min_validation_mse = 1.0e30
        self.min_validation_mse_epoch_number = -1
        self.prev_train_err = 1.0e30
        self.workflow = None
        self.fnme = None
        self.fnmeWb = None
        self.t1 = None
        self.epoch_metrics = [None, None, None]
        self.just_snapshotted = [0]
        self.snapshot_time = [0]
        self.threshold_ok = 0.0005
        self.sample_output = None
        self.sample_input = None
        self.sample_target = None

    def initialize(self):
        if (self.minibatch_metrics != None and
            self.minibatch_metrics.v != None):
            for i in range(0, len(self.epoch_metrics)):
                self.epoch_metrics[i] = (
                    numpy.zeros_like(self.minibatch_metrics.v))
        self.sample_output = numpy.zeros_like(
            self.workflow.forward[-1].output.batch[0])
        self.sample_input = numpy.zeros_like(
            self.workflow.forward[0].input.batch[0])
        self.sample_target = numpy.zeros_like(
            self.workflow.ev.target.batch[0])

    def run(self):
        if self.t1 == None:
            self.t1 = time.time()
        self.complete[0] = 0
        self.epoch_ended[0] = 0

        minibatch_class = self.minibatch_class[0]

        if self.minibatch_last[0]:
            self.minibatch_metrics.sync()
            self.epoch_min_mse[minibatch_class] = (
                min(self.minibatch_metrics.v[0] /
                    self.class_samples[minibatch_class],
                self.epoch_min_mse[minibatch_class]))

            self.minibatch_n_err.sync()
            self.n_err[minibatch_class] = self.minibatch_n_err.v[0]

            # Compute error in percents
            if self.class_samples[minibatch_class]:
                self.n_err_pt[minibatch_class] = (100.0 *
                    self.n_err[minibatch_class] /
                    self.class_samples[minibatch_class])

        # Check skip gradient descent or not
        if self.minibatch_class[0] < 2:
            self.gd_skip[0] = 1
        else:
            self.gd_skip[0] = 0

        if self.minibatch_last[0]:
            self.epoch_metrics[minibatch_class][:] = (
                self.minibatch_metrics.v[:])
            self.epoch_metrics[minibatch_class][0] = (
                self.epoch_metrics[minibatch_class][0] /
                self.class_samples[minibatch_class])

            # Test and Validation sets processed
            if self.minibatch_class[0] >= 1:
                if self.just_snapshotted[0]:
                    self.just_snapshotted[0] = 0
                if (self.epoch_min_mse[minibatch_class] <
                    self.min_validation_mse):
                    self.min_validation_mse = self.epoch_min_mse[
                        minibatch_class]
                    self.min_validation_mse_epoch_number = self.epoch_number[0]
                    global this_dir
                    if self.fnme != None:
                        try:
                            os.unlink(self.fnme)
                        except FileNotFoundError:
                            pass
                    if self.fnmeWb != None:
                        try:
                            os.unlink(self.fnmeWb)
                        except FileNotFoundError:
                            pass
                    self.fnme = "%s/kanji_%.6f.pickle" % \
                        (config.snapshot_dir,
                         self.epoch_metrics[minibatch_class][0])
                    self.log().info("Snapshotting to %s" % (self.fnme,))
                    fout = open(self.fnme, "wb")
                    pickle.dump(self.workflow, fout)
                    fout.close()
                    self.fnmeWb = "%s/kanji_%.6f_Wb.pickle" % \
                        (config.snapshot_dir,
                         self.epoch_metrics[minibatch_class][0])
                    self.log().info("Exporting weights to %s" % (self.fnmeWb,))
                    fout = open(self.fnmeWb, "wb")
                    weights = []
                    bias = []
                    for forward in self.workflow.forward:
                        forward.weights.sync(read_only=True)
                        forward.bias.sync(read_only=True)
                        weights.append(forward.weights.v)
                        bias.append(forward.bias.v)
                        self.log().info("%f %f %f %f" % (
                            forward.weights.v.min(), forward.weights.v.max(),
                            forward.bias.v.min(), forward.bias.v.max()))
                    pickle.dump((weights, bias), fout)
                    fout.close()
                    self.just_snapshotted[0] = 1
                    self.snapshot_time[0] = time.time()
                # Stop condition
                if self.epoch_number[0] - \
                   self.min_validation_mse_epoch_number > \
                   self.fail_iterations[0]:
                    self.complete[0] = 1
                #self.workflow.ev.threshold_skip = \
                #    self.epoch_metrics[minibatch_class][0]

            # Print some statistics
            t2 = time.time()
            self.log().info(
                "Epoch %d Class %d AvgMSE %.6f Greater%.3f %d (%.2f%%) "
                "MaxMSE %.6f MinMSE %.2e in %.2f sec" %
                (self.epoch_number[0], minibatch_class,
                 self.epoch_metrics[minibatch_class][0],
                 self.threshold_ok,
                 self.n_err[minibatch_class],
                 self.n_err_pt[minibatch_class],
                 self.epoch_metrics[minibatch_class][1],
                 self.epoch_metrics[minibatch_class][2],
                 t2 - self.t1))
            self.t1 = t2

            # Training set processed
            if self.minibatch_class[0] == 2:
                """
                this_train_err = self.epoch_metrics[2][0]
                if self.prev_train_err:
                    k = this_train_err / self.prev_train_err
                else:
                    k = 1.0
                if k < 1.04:
                    ak = 1.05
                else:
                    ak = 0.7
                self.prev_train_err = this_train_err
                for gd in self.workflow.gd:
                    gd.global_alpha = max(min(ak * gd.global_alpha, 0.99999),
                                          0.00001)
                self.log().info("new global_alpha: %.4f" % \
                      (self.workflow.gd[0].global_alpha, ))
                """
                self.epoch_ended[0] = 1
                self.epoch_number[0] += 1
                # Reset n_err
                for i in range(0, len(self.n_err)):
                    self.n_err[i] = 0
                # Sync weights
                #self.weights_to_sync.sync(read_only=True)
                self.workflow.forward[0].input.sync(read_only=True)
                self.workflow.forward[-1].output.sync(read_only=True)
                self.workflow.ev.target.sync(read_only=True)
                self.sample_output[:] = \
                    self.workflow.forward[-1].output.batch[0][:]
                self.sample_input[:] = \
                    self.workflow.forward[0].input.batch[0][:]
                self.sample_target[:] = \
                    self.workflow.ev.target.batch[0][:]

            # Reset statistics per class
            self.minibatch_n_err.v[:] = 0
            self.minibatch_n_err.update()
            if (self.minibatch_metrics != None and
                self.minibatch_metrics.v != None):
                self.minibatch_metrics.v[:] = 0
                self.minibatch_metrics.v[2] = 1.0e30
                self.minibatch_metrics.update()


class Workflow(units.OpenCLUnit):
    """Sample workflow for MNIST dataset.

    Attributes:
        start_point: start point.
        rpt: repeater.
        loader: loader.
        forward: list of all-to-all forward units.
        ev: evaluator softmax.
        stat: stat collector.
        decision: Decision.
        gd: list of gradient descent units.
    """
    def __init__(self, layers=None, device=None, unpickling=None):
        super(Workflow, self).__init__(device=device, unpickling=unpickling)
        if unpickling:
            return
        self.start_point = units.Unit()

        self.rpt = units.Repeater()
        self.rpt.link_from(self.start_point)

        self.loader = Loader()
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward = []
        for i in range(0, len(layers)):
            if not i:
                amp = None
            else:
                amp = 9.0 / 1.7159 / layers[i - 1]
            #amp = 0.05
            aa = all2all.All2AllTanh([layers[i]], device=device,
                                     weights_amplitude=amp)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add Image Saver unit
        """
        self.image_saver = ImageSaverAE()
        self.image_saver.link_from(self.forward[-1])
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.forward[-1].output
        self.image_saver.indexes = self.loader.minibatch_indexes
        self.image_saver.labels = self.loader.minibatch_labels
        self.image_saver.minibatch_class = self.loader.minibatch_class
        self.image_saver.minibatch_size = self.loader.minibatch_size
        """

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorMSE(device=device, threshold_ok=0.005)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.target = self.loader.minibatch_target
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = Decision()
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err_skipped
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        #self.image_saver.this_save_time = self.decision.snapshot_time
        #self.image_saver.gate_skip = self.decision.just_snapshotted
        #self.image_saver.gate_skip_not = [1]

        # Add gradient descent units
        self.gd = list(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDTanh(device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.ev.effective_batch_size

        self.gd[-2] = gd.GDTanh(device=device)
        self.gd[-2].link_from(self.gd[-1])
        self.gd[-2].err_y = self.gd[-1].err_h
        self.gd[-2].y = self.forward[-2].output
        self.gd[-2].h = self.forward[-2].input
        self.gd[-2].weights = self.forward[-2].weights
        self.gd[-2].bias = self.forward[-2].bias
        self.gd[-2].gate_skip = self.decision.gd_skip
        self.gd[-2].batch_size = self.ev.effective_batch_size

        self.rpt.link_from(self.gd[0])

        self.end_point = units.EndPoint()
        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["", "", "k-"]  # ["r-", "b-", "k-"]
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt.append(plotters.SimplePlotter(figure_label="mse",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = self.decision.epoch_ended
            self.plt[-1].gate_block_not = [1]
        # Matrix plotter
        #self.decision.weights_to_sync = self.gd[0].weights
        #self.decision.output_to_sync = self.forward[-1].output
        #self.plt_mx = plotters.Weights2D(figure_label="First Layer Weights")
        #self.plt_mx.input = self.decision.weights_to_sync
        #self.plt_mx.input_field = "v"
        #self.plt_mx.link_from(self.decision)
        #self.plt_mx.gate_block = self.decision.epoch_ended
        #self.plt_mx.gate_block_not = [1]
        # Max plotter
        self.plt_max = []
        styles = ["", "", "k--"]  # ["r--", "b--", "k--"]
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_max.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.decision)
            self.plt_max[-1].gate_block = self.decision.epoch_ended
            self.plt_max[-1].gate_block_not = [1]
        # Min plotter
        self.plt_min = []
        styles = ["", "", "k:"]  # ["r:", "b:", "k:"]
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_min.append(plotters.SimplePlotter(figure_label="mse",
                                                       plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.decision)
            self.plt_min[-1].gate_block = self.decision.epoch_ended
            self.plt_min[-1].gate_block_not = [1]
        # Image plotter
        self.plt_img = plotters.Image3(figure_label="output sample")
        self.plt_img.input = self.decision
        self.plt_img.input_field = "sample_input"
        self.plt_img.input_field2 = "sample_output"
        self.plt_img.input_field3 = "sample_target"
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_block = self.decision.epoch_ended
        self.plt_img.gate_block_not = [1]

    def initialize(self, device, threshold_ok, threshold_skip,
                   global_alpha, global_lambda,
                   minibatch_maxsize):
        for gd in self.gd:
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
            gd.device = device
        for forward in self.forward:
            forward.device = device
        self.ev.device = device
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        self.decision.threshold_ok = threshold_ok
        self.ev.threshold_ok = threshold_ok
        self.ev.threshold_skip = threshold_skip
        retval = self.start_point.initialize_dependent()
        if retval:
            return retval

    def run(self, weights, bias):
        if weights != None:
            for i, forward in enumerate(self.forward):
                forward.weights.v[:] = weights[i][:]
                forward.weights.update()
        if bias != None:
            for i, forward in enumerate(self.forward):
                forward.bias.v[:] = bias[i][:]
                forward.bias.update()
        retval = self.start_point.run_dependent()
        if retval:
            return retval
        self.end_point.wait()


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    """This is a test for correctness of a particular trained 2-layer network.
    fin = open("mnist.pickle", "rb")
    w = pickle.load(fin)
    fin.close()

    fout = open("w100.txt", "w")
    weights = w.forward[0].weights.v
    for row in weights:
        fout.write(" ".join("%.6f" % (x, ) for x in row))
        fout.write("\n")
    fout.close()
    fout = open("b100.txt", "w")
    bias = w.forward[0].bias.v
    fout.write(" ".join("%.6f" % (x, ) for x in bias))
    fout.write("\n")
    fout.close()

    a = w.loader.original_data.reshape(70000, 784)[0:10000]
    b = weights.transpose()
    c = numpy.zeros([10000, 100], dtype=a.dtype)
    numpy.dot(a, b, c)
    c[:] += bias
    c *= 0.6666
    numpy.tanh(c, c)
    c *= 1.7159

    fout = open("w10.txt", "w")
    weights = w.forward[1].weights.v
    for row in weights:
        fout.write(" ".join("%.6f" % (x, ) for x in row))
        fout.write("\n")
    fout.close()
    fout = open("b10.txt", "w")
    bias = w.forward[1].bias.v
    fout.write(" ".join("%.6f" % (x, ) for x in bias))
    fout.write("\n")
    fout.close()

    a = c
    b = weights.transpose()
    c = numpy.zeros([10000, 10], dtype=a.dtype)
    numpy.dot(a, b, c)
    c[:] += bias

    labels = w.loader.original_labels[0:10000]
    n_ok = 0
    for i in range(0, 10000):
        im = numpy.argmax(c[i])
        if im == labels[i]:
            n_ok += 1
    self.log().info("%d errors" % (10000 - n_ok, ))

    self.log().info("Done")
    sys.exit(0)
    """

    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir,),
                                    numpy.int32, 1024))
    #rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 524288))
    try:
        cl = opencl.DeviceList()
        device = cl.get_device()
        layers = [1998, 1485, 24 * 24]
        fnme = "%s/kanji.pickle" % (this_dir,)
        fin = None
        try:
            fin = open(fnme, "rb")
        except IOError:
            pass
        weights = None
        bias = None
        if fin != None:
            w = pickle.load(fin)
            fin.close()
            if type(w) == list:
                weights = w[0]
                bias = w[1]
                fin = None
            else:
                logging.info("Weights and bias ranges per layer are:")
                for forward in w.forward:
                    logging.info("%f %f %f %f" % (
                        forward.weights.v.min(), forward.weights.v.max(),
                        forward.bias.v.min(), forward.bias.v.max()))
                w.decision.just_snapshotted[0] = 1
        if fin == None:
            w = Workflow(layers=layers, device=device)
    except KeyboardInterrupt:
        return
    try:
        N = 3
        for i in range(0, N):
            w = Workflow(layers=layers, device=device)
            w.initialize(threshold_ok=0.004, threshold_skip=0.0,
                         global_alpha=0.001, global_lambda=0.00005,
                         minibatch_maxsize=594, device=device)
            w.run(weights=weights, bias=bias)
            weights = []
            bias = []
            # Add another layer
            if i < N - 1:
                logging.info("Adding another layer")
                for i in range(0, len(w.forward) - 1):
                    w.forward[i].weights.sync(read_only=True)
                    w.forward[i].bias.sync(read_only=True)
                    weights.append(w.forward[i].weights.v)
                    bias.append(w.forward[i].bias.v)
                layers.insert(len(layers) - 1, layers[-2])
    except KeyboardInterrupt:
        w.gd[-1].gate_block = [1]
    logging.info("Will snapshot after 15 seconds...")
    time.sleep(5)
    logging.info("Will snapshot after 10 seconds...")
    time.sleep(5)
    logging.info("Will snapshot after 5 seconds...")
    time.sleep(5)
    fnme = "%s/kanji.pickle" % (config.snapshot_dir,)
    logging.info("Snapshotting to %s" % (fnme,))
    fout = open(fnme, "wb")
    pickle.dump(w, fout)
    fout.close()

    plotters.Graphics().wait_finish()
    logging.info("End of job")


if __name__ == "__main__":
    main()
    sys.exit()
