#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

File for MNIST dataset.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s/../src" % (this_dir, ))


import units
import formats
import struct
import error
import numpy
import config
import rnd
import opencl
import plotters


class MNISTLoader(units.Unit):
    """Loads MNIST data and provides mini-batch output interface.

    Attributes:
        rnd: rnd.Rand().

        minibatch_data: MNIST images scaled to [-1, 1].
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

        original_data: original MNIST images scaled to [-1, 1] as single batch.
        original_labels: original MNIST labels as single batch.
    """
    def __init__(self, classes=[0, 10000, 60000], minibatch_max_size=60,
                 rnd=rnd.default, unpickling=0):
        """Constructor.

        Parameters:
            classes: [test, validation, train],
                ints - in samples,
                floats - relative from (0 to 1).
            minibatch_size: minibatch max size.
        """
        super(MNISTLoader, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.rnd = [rnd]

        self.minibatch_data = formats.Batch()
        self.minibatch_indexes = formats.Labels(70000)
        self.minibatch_labels = formats.Labels(10)

        self.minibatch_class = [0]
        self.minibatch_last = [0]

        self.total_samples = [70000]
        self.class_samples = classes.copy()
        if type(self.class_samples[2]) == float:
            smm = 0
            for i in range(0, len(self.class_samples) - 1):
                self.class_samples[i] = int(
                numpy.round(self.total_samples[0] * self.class_samples[i]))
                smm += self.class_samples[i]
            self.class_samples[-1] = self.total_samples[0] - smm
        self.minibatch_offs = [self.total_samples[0]]
        self.minibatch_size = [0]
        self.minibatch_maxsize = [minibatch_max_size]
        self.nextclass_offs = [0, 0, 0]
        offs = 0
        for i in range(0, len(self.class_samples)):
            offs += self.class_samples[i]
            self.nextclass_offs[i] = offs
        if self.nextclass_offs[-1] != self.total_samples[0]:
            raise error.ErrBadFormat("Sum of class samples (%d) differs from "
                "total number of samples (%d)" % (self.nextclass_offs[-1],
                                                  self.total_samples))

        self.original_data = None
        self.original_labels = None

        self.shuffled_indexes = None

    def load_original(self, offs, labels_count, labels_fnme, images_fnme):
        """Loads data from original MNIST files.
        """
        print("Loading from original MNIST files...")

        # Reading labels:
        fin = open(labels_fnme, "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2049:
            raise error.ErrBadFormat("Wrong header in train-labels")

        n_labels, = struct.unpack(">i", fin.read(4))
        if n_labels != labels_count:
            raise error.ErrBadFormat("Wrong number of labels in train-labels")

        arr = numpy.fromfile(fin, dtype=numpy.byte, count=n_labels)
        if arr.size != n_labels:
            raise error.ErrBadFormat("EOF reached while reading labels from "
                                     "train-labels")
        self.original_labels[offs:offs + labels_count] = arr[:]
        if self.original_labels.min() != 0 or self.original_labels.max() != 9:
            raise error.ErrBadFormat("Wrong labels range in train-labels.")

        fin.close()

        # Reading images:
        fin = open(images_fnme, "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2051:
            raise error.ErrBadFormat("Wrong header in train-images")

        n_images, = struct.unpack(">i", fin.read(4))
        if n_images != n_labels:
            raise error.ErrBadFormat("Wrong number of images in train-images")

        n_rows, n_cols = struct.unpack(">2i", fin.read(8))
        if n_rows != 28 or n_cols != 28:
            raise error.ErrBadFormat("Wrong images size in train-images, "
                                     "should be 28*28")

        # 0 - white, 255 - black
        pixels = numpy.fromfile(fin, dtype=numpy.ubyte,
                                count=n_images * n_rows * n_cols)
        if pixels.shape[0] != n_images * n_rows * n_cols:
            raise error.ErrBadFormat("EOF reached while reading images "
                                     "from train-images")

        fin.close()

        # Transforming images into float arrays and normalizing to [-1, 1]:
        images = pixels.astype(config.dtypes[config.dtype]).\
            reshape(n_images, n_rows, n_cols)
        print("Original range: [%.1f, %.1f]" % (images.min(), images.max()))
        for image in images:
            vle_min = image.min()
            vle_max = image.max()
            image += -vle_min
            image *= 2.0 / (vle_max - vle_min)
            image += -1.0
        print("Range after normalization: [%.1f, %.1f]" % (images.min(),
                                                           images.max()))
        self.original_data[offs:offs + n_images] = images[:]
        print("Done")

    def initialize(self):
        """Here we will load MNIST data.
        """
        if not self.original_labels or self.original_labels.size < 70000:
            self.original_labels = numpy.zeros([70000], dtype=numpy.int8)
        if not self.original_data or self.original_data.shape[0] < 70000:
            self.original_data = numpy.zeros([70000, 28, 28],
                                             dtype=config.dtypes[config.dtype])
        if not self.shuffled_indexes or self.shuffled_indexes.size < 70000:
            self.shuffled_indexes = numpy.arange(70000, dtype=numpy.int32)

        global this_dir
        self.load_original(0, 10000,
                           "%s/MNIST/t10k-labels.idx1-ubyte" % (this_dir, ),
                           "%s/MNIST/t10k-images.idx3-ubyte" % (this_dir, ))
        self.load_original(10000, 60000,
                           "%s/MNIST/train-labels.idx1-ubyte" % (this_dir, ),
                           "%s/MNIST/train-images.idx3-ubyte" % (this_dir, ))

        self.minibatch_data.batch = numpy.zeros(
            [self.minibatch_maxsize[0], 28, 28],
            dtype=config.dtypes[config.dtype])
        self.minibatch_labels.batch = numpy.zeros(
            [self.minibatch_maxsize[0]], dtype=numpy.int8)
        self.minibatch_indexes.batch = numpy.zeros(
            [self.minibatch_maxsize[0]], dtype=numpy.int32)

        if self.class_samples[0]:
            self.shuffle_validation_train()
        else:
            self.shuffle_train()

    def shuffle_validation_train(self):
        """Shuffles original train dataset
            and allocates 10000 for validation,
            so the layout will be:
                0:10000: test,
                10000:20000: validation,
                20000:70000: train.
        """
        self.rnd[0].shuffle(self.shuffled_indexes[self.nextclass_offs[0]:\
                                                  self.nextclass_offs[2]])

    def shuffle_train(self):
        """Shuffles used train dataset
            so the layout will be:
                0:10000: test,
                10000:20000: validation,
                20000:70000: randomized train.
        """
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

        # Sync from GPU if neccessary.
        self.minibatch_data.sync()

        # Fill minibatch data labels and indexes according to current shuffle.
        idxs = self.minibatch_indexes.batch
        idxs[0:minibatch_size] = self.shuffled_indexes[self.minibatch_offs[0]:\
            self.minibatch_offs[0] + minibatch_size]

        self.minibatch_labels.batch[0:minibatch_size] = \
            self.original_labels[idxs[0:minibatch_size]]

        self.minibatch_data.batch[0:minibatch_size] = \
            self.original_data[idxs[0:minibatch_size]]

        # Fill excessive indexes.
        if minibatch_size < self.minibatch_maxsize[0]:
            self.minibatch_data.batch[minibatch_size:] = 0.0
            self.minibatch_labels.batch[minibatch_size:] = -1
            self.minibatch_indexes.batch[minibatch_size:] = -1

        # Set update flag for GPU operation.
        self.minibatch_data.update()
        self.minibatch_labels.update()
        self.minibatch_indexes.update()

        if __debug__:
            print("%s in %.2f sec" % (self.__class__.__name__,
                                      time.time() - t1))


import all2all
import evaluator
import gd


class Decision(units.Unit):
    """Decides on the learning behavior.

    Attributes:
        complete: completed.
        minibatch_class: current minibatch class.
        minibatch_last: if current minibatch is last in it's class.
        gd_skip: skip gradient descent or not.
        epoch_number: epoch number.
        epoch_min_err: minimum number of errors by class per epoch.
        n_err: current number of errors per class.
        minibatch_n_err: number of errors for minibatch.
        n_err_pt: n_err in percents.
        class_samples: number of samples per class.
        epoch_ended: if an epoch has ended.
        fail_iterations: number of consequent iterations with non-decreased
            validation error.
    """
    def __init__(self, fail_iterations=500, unpickling=0):
        super(Decision, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.complete = [0]
        self.minibatch_class = None  # [0]
        self.minibatch_last = None  # [0]
        self.gd_skip = [0]
        self.epoch_number = [0]
        self.epoch_min_err = [1.0e30, 1.0e30, 1.0e30]
        self.n_err = [0, 0, 0]
        self.minibatch_n_err = None  # [0]
        self.fail_iterations = [fail_iterations]
        self.epoch_ended = [0]
        self.n_err_pt = [100.0, 100.0, 100.0]
        self.class_samples = None  # [0, 0, 0]
        self.min_validation_err = 1.0e30
        self.min_validation_err_epoch_number = -1
        self.workflow = None
        self.fnme = None

    def run(self):
        self.complete[0] = 0
        self.epoch_ended[0] = 0

        minibatch_class = self.minibatch_class[0]
        self.n_err[minibatch_class] += self.minibatch_n_err[0]

        if self.minibatch_last[0]:
            self.epoch_min_err[minibatch_class] = \
                min(self.n_err[minibatch_class],
                    self.epoch_min_err[minibatch_class])

        # Compute errors in percents
        for i in range(0, len(self.n_err_pt)):
            if self.class_samples[i]:
                self.n_err_pt[i] = self.n_err[i] / self.class_samples[i]
                self.n_err_pt[i] *= 100.0

        # Check skip gradient descent or not
        if self.minibatch_class[0] < 2:
            self.gd_skip[0] = 1
        else:
            self.gd_skip[0] = 0

        if self.minibatch_last[0]:
            # Test and Validation sets processed
            if self.minibatch_class[0] == 1:
                if self.epoch_min_err[1] < self.min_validation_err:
                    self.min_validation_err = self.epoch_min_err[1]
                    self.min_validation_err_epoch_number = self.epoch_number[0]
                    if self.n_err_pt[1] < 5.0:
                        global this_dir
                        if self.fnme != None:
                            os.unlink(self.fnme)
                        self.fnme = "%s/mnist.%.2f.pickle" % \
                            (this_dir, self.n_err_pt[1])
                        print("Snapshotting to %s" % (self.fnme, ))
                        fout = open(self.fnme, "wb")
                        pickle.dump(self.workflow, fout)
                        fout.close()
                # Stop condition
                if self.epoch_number[0] - \
                   self.min_validation_err_epoch_number > \
                   self.fail_iterations[0]:
                    self.complete[0] = 1

            # Print some statistics
            print("Epoch %d Class %d Errors %d" % \
                  (self.epoch_number[0], self.minibatch_class[0],
                   self.n_err[self.minibatch_class[0]]))

            # Training set processed
            if self.minibatch_class[0] == 2:
                self.epoch_ended[0] = 1
                self.epoch_number[0] += 1
                # Reset n_err
                for i in range(0, len(self.n_err)):
                    self.n_err[i] = 0


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

        self.loader = MNISTLoader()
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward = []
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh([layers[i]], device=device)
            else:
                aa = all2all.All2AllSoftmax([layers[i]], device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.labels = self.loader.minibatch_labels

        # Add decision unit
        self.decision = Decision()
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add gradient descent units
        self.gd = list(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
        self.rpt.link_from(self.gd[0])

        self.end_point = units.EndPoint()
        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # Plotter here
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotters.SimplePlotter(device=device,
                            figure_label="num errors",
                            plot_style=styles[i]))
            self.plt[-1].input = self.decision.n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = self.decision.epoch_ended
            self.plt[-1].gate_block_not = [1]

    def initialize(self):
        retval = self.start_point.initialize_dependent()
        if retval:
            return retval

    def run(self, threshold, threshold_low, global_alpha, global_lambda):
        self.ev.threshold = threshold
        self.ev.threshold_low = threshold_low
        for gd in self.gd:
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
        retval = self.start_point.run_dependent()
        if retval:
            return retval
        self.end_point.wait()


import inline
import pickle
import time


def main():
    global this_dir
    #rnd.default.seed(numpy.fromfile("%s/scripts/seed" % (this_dir, ),
    #                                numpy.int32, 1024))
    rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    unistd = inline.Inline()
    unistd.sources.append("#include <unistd.h>")
    unistd.function_descriptions = {"_exit": "iv"}
    unistd.compile()
    try:
        cl = opencl.DeviceList()
        device = cl.get_device()
        w = Workflow(layers=[100, 10], device=device)
        w.initialize()
    except KeyboardInterrupt:
        unistd.execute("_exit", 0)
    try:
        w.run(threshold=1.0, threshold_low=1.0,
              global_alpha=0.1, global_lambda=0.000)
    except KeyboardInterrupt:
        w.gd[-1].gate_block = [1]
    print("Will snapshot after 15 seconds...")
    time.sleep(5)
    print("Will snapshot after 10 seconds...")
    time.sleep(5)
    print("Will snapshot after 5 seconds...")
    time.sleep(5)
    fnme = "%s/mnist.pickle" % (this_dir, )
    print("Snapshotting to %s" % (fnme, ))
    fout = open(fnme, "wb")
    pickle.dump(w, fout)
    fout.close()
    #print("Will now exit")
    #unistd.execute("_exit", 0)
    plotters.Graphics().wait_finish()


if __name__ == "__main__":
    main()
