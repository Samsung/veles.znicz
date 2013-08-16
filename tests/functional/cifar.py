#!/usr/bin/python3.3 -O
"""
Created on Jul 3, 2013

File for channels recognition.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import os
import logging


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import units
import formats
import numpy
import config
import rnd
import opencl
import plotters
import glob
import pickle
import time
import scipy.ndimage
import loader


class Loader(loader.FullBatchLoader):
    """Loads Cifar dataset.
    """
    def load_data(self):
        """Here we will load data.
        """
        n_classes = 10
        self.original_data = numpy.zeros([60000, 3, 32, 32],
                                         dtype=config.dtypes[config.dtype])
        self.original_labels = numpy.zeros(60000,
            dtype=config.itypes[config.get_itype_from_size(n_classes)])

        # Load Validation
        fin = open("%s/cifar/10/test_batch" % (config.test_dataset_root),
                   "rb")
        u = pickle._Unpickler(fin)
        u.encoding = 'latin1'
        vle = u.load()
        fin.close()
        self.original_data[:10000] = vle["data"].reshape(10000, 3, 32, 32)[:]
        self.original_labels[:10000] = vle["labels"][:]

        # Load Train
        for i in range(1, 6):
            fin = open("%s/cifar/10/data_batch_%d" % (config.test_dataset_root,
                       i), "rb")
            u = pickle._Unpickler(fin)
            u.encoding = 'latin1'
            vle = u.load()
            fin.close()
            self.original_data[i * 10000: (i + 1) * 10000] = \
                vle["data"].reshape(10000, 3, 32, 32)[:]
            self.original_labels[i * 10000: (i + 1) * 10000] = vle["labels"][:]

        self.class_samples[0] = 0
        self.nextclass_offs[0] = 0
        self.class_samples[1] = 10000
        self.nextclass_offs[1] = 10000
        self.class_samples[2] = 50000
        self.nextclass_offs[2] = 60000

        self.total_samples[0] = self.original_data.shape[0]

        for sample in self.original_data:
            formats.normalize(sample)


import all2all
import evaluator
import gd
import scipy.misc


class ImageSaver(units.Unit):
    """Saves input and output side by side to png for AutoEncoder.

    Attributes:
        out_dirs: output directories by minibatch_class where to save png.
        input: batch with input samples.
        output: batch with corresponding output samples.
        indexes: sample indexes.
        labels: sample labels.
    """
    def __init__(self, out_dirs=[".", ".", "."]):
        super(ImageSaver, self).__init__()
        self.out_dirs = out_dirs
        self.input = None  # formats.Vector()
        self.output = None  # formats.Vector()
        self.indexes = None  # formats.Vector()
        self.labels = None  # formats.Vector()
        self.minibatch_class = None  # [0]
        self.minibatch_size = None  # [0]
        self.snapshot_date = None
        self.last_snapshot_date = 0
        self.max_idx = None

    def run(self):
        self.input.sync()
        self.output.sync()
        self.max_idx.sync()
        self.indexes.sync()
        self.labels.sync()
        dirnme = self.out_dirs[self.minibatch_class[0]]
        if self.snapshot_date[0] != self.last_snapshot_date:
            self.last_snapshot_date = self.snapshot_date[0]
            files = glob.glob("%s/*.png" % (dirnme))
            for file in files:
                try:
                    os.unlink(file)
                except OSError:
                    pass
            del files
        for i in range(0, self.minibatch_size[0]):
            x = self.input.v[i]
            y = self.output.v[i]
            idx = self.indexes.v[i]
            lbl = self.labels.v[i]
            im = self.max_idx[i]
            if im == lbl:
                continue
            fnme = "%s/%d_as_%d.%.0fpt.%d.png" % (dirnme, lbl, im, y[im], idx)
            scipy.misc.imsave(fnme, x)


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
        confusion_matrix: confusion matrix.
    """
    def __init__(self, fail_iterations=100):
        super(Decision, self).__init__()
        self.complete = [0]
        self.minibatch_class = None  # [0]
        self.minibatch_last = None  # [0]
        self.gd_skip = [0]
        self.epoch_number = [0]
        self.epoch_min_err = [1.0e30, 1.0e30, 1.0e30]
        self.n_err = [0, 0, 0]
        self.minibatch_n_err = None  # formats.Vector()
        self.minibatch_confusion_matrix = None  # formats.Vector()
        self.fail_iterations = [fail_iterations]
        self.epoch_ended = [0]
        self.n_err_pt = [100.0, 100.0, 100.0]
        self.class_samples = None  # [0, 0, 0]
        self.min_validation_err = 1.0e30
        self.min_validation_err_epoch_number = -1
        self.prev_train_err = 1.0e30
        self.workflow = None
        self.fnme = None
        self.t1 = None
        self.confusion_matrixes = [None, None, None]
        self.snapshot_date = [0]
        self.just_snapshotted = [0]
        self.weights_to_sync = None

    def init_unpickled(self):
        super(Decision, self).init_unpickled()
        self.epoch_min_err = [1.0e30, 1.0e30, 1.0e30]
        self.n_err_pt = [100.0, 100.0, 100.0]
        self.n_err = [0, 0, 0]

    def initialize(self):
        if (self.minibatch_confusion_matrix == None or
            self.minibatch_confusion_matrix.v == None):
            return
        for i in range(0, len(self.confusion_matrixes)):
            self.confusion_matrixes[i] = (
                numpy.zeros_like(self.minibatch_confusion_matrix.v))

    def run(self):
        if self.t1 == None:
            self.t1 = time.time()
        self.complete[0] = 0
        self.epoch_ended[0] = 0

        minibatch_class = self.minibatch_class[0]

        if self.minibatch_last[0]:
            self.minibatch_n_err.sync()
            self.n_err[minibatch_class] += self.minibatch_n_err.v[0]
            self.epoch_min_err[minibatch_class] = \
                min(self.n_err[minibatch_class],
                    self.epoch_min_err[minibatch_class])
            # Compute error in percents
            if self.class_samples[minibatch_class]:
                self.n_err_pt[minibatch_class] = (self.n_err[minibatch_class] /
                    self.class_samples[minibatch_class])
                self.n_err_pt[minibatch_class] *= 100.0

        # Check skip gradient descent or not
        if self.minibatch_class[0] < 2:
            self.gd_skip[0] = 1
        else:
            self.gd_skip[0] = 0

        if self.minibatch_last[0]:
            if (self.minibatch_confusion_matrix != None and
                self.minibatch_confusion_matrix.v != None):
                self.minibatch_confusion_matrix.sync()
                self.confusion_matrixes[minibatch_class][:] = (
                    self.minibatch_confusion_matrix.v[:])

            # Test and Validation sets processed
            if ((self.class_samples[1] and minibatch_class == 1) or
                (not self.class_samples[1] and minibatch_class == 2)):
                if self.just_snapshotted[0]:
                    self.just_snapshotted[0] = 0
                if (self.epoch_min_err[minibatch_class] <
                    self.min_validation_err):
                    self.min_validation_err = \
                        self.epoch_min_err[minibatch_class]
                    self.min_validation_err_epoch_number = self.epoch_number[0]
                    if self.n_err_pt[minibatch_class] < 50.0:
                        global this_dir
                        if self.fnme != None:
                            try:
                                os.unlink(self.fnme)
                            except FileNotFoundError:
                                pass
                        self.fnme = "%s/channels_%.2f.pickle" % \
                            (this_dir, self.n_err_pt[minibatch_class])
                        self.log().info(
                            "                                        "
                            "Snapshotting to %s" % (self.fnme))
                        fout = open(self.fnme, "wb")
                        pickle.dump(self.workflow, fout)
                        fout.close()
                        self.just_snapshotted[0] = 1
                        self.snapshot_date[0] = time.time()
                # Stop condition
                if self.epoch_number[0] - \
                   self.min_validation_err_epoch_number > \
                   self.fail_iterations[0]:
                    self.complete[0] = 1

            # Print some statistics
            t2 = time.time()
            self.log().info("Epoch %d Class %d Errors %d in %.2f sec" % \
                  (self.epoch_number[0], minibatch_class,
                   self.n_err[minibatch_class],
                   t2 - self.t1))
            self.t1 = t2

            # Training set processed
            if self.minibatch_class[0] == 2:
                """
                this_train_err = self.n_err[2]
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
                    gd.global_alpha = max(min(ak * gd.global_alpha, 0.9999),
                                          0.0001)
                self.log().info("new global_alpha: %.4f" % \
                      (self.workflow.gd[0].global_alpha))
                """
                self.epoch_ended[0] = 1
                self.epoch_number[0] += 1
                # Reset n_err
                for i in range(0, len(self.n_err)):
                    self.n_err[i] = 0
                # Sync weights
                if self.weights_to_sync != None:
                    self.weights_to_sync.sync()

            # Reset statistics per class
            self.minibatch_n_err.v[:] = 0
            self.minibatch_n_err.update()
            if (self.minibatch_confusion_matrix != None and
                self.minibatch_confusion_matrix.v != None):
                self.minibatch_confusion_matrix.v[:] = 0
                self.minibatch_confusion_matrix.update()


class Workflow(units.OpenCLUnit):
    """Sample workflow.

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
    def __init__(self, layers=None, device=None):
        super(Workflow, self).__init__(device=device)
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
            # amp = 0.05
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh([layers[i]], device=device,
                                         weights_amplitude=amp)
            else:
                aa = all2all.All2AllSoftmax([layers[i]], device=device,
                                            weights_amplitude=amp)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add Image Saver unit
        self.image_saver = ImageSaver(["/data/veles/cifar/tmpimg/test",
            "/data/veles/cifar/tmpimg/validation",
            "/data/veles/cifar/tmpimg/train"])
        self.image_saver.link_from(self.forward[-1])
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.forward[-1].output
        self.image_saver.max_idx = self.forward[-1].max_idx
        self.image_saver.indexes = self.loader.minibatch_indexes
        self.image_saver.labels = self.loader.minibatch_labels
        self.image_saver.minibatch_class = self.loader.minibatch_class
        self.image_saver.minibatch_size = self.loader.minibatch_size

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(device=device)
        self.ev.link_from(self.image_saver)
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_idx = self.forward[-1].max_idx
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = Decision()
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err_skipped
        self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        self.image_saver.gate_skip = self.decision.just_snapshotted
        self.image_saver.gate_skip_not = [1]
        self.image_saver.snapshot_date = self.decision.snapshot_date

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

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotters.SimplePlotter(figure_label="num errors",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = self.decision.epoch_ended
            self.plt[-1].gate_block_not = [1]
        # Matrix plotter
        # """
        self.decision.weights_to_sync = self.gd[0].weights
        self.plt_w = plotters.Weights2D(figure_label="First Layer Weights",
                                        limit=25)
        self.plt_w.input = self.decision.weights_to_sync
        self.plt_w.get_shape_from = self.forward[0].input
        self.plt_w.input_field = "v"
        self.plt_w.link_from(self.decision)
        self.plt_w.gate_block = self.decision.epoch_ended
        self.plt_w.gate_block_not = [1]
        # """
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(0, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotters.MatrixPlotter(
                figure_label=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.plt[-1])
            self.plt_mx[-1].gate_block = self.decision.epoch_ended
            self.plt_mx[-1].gate_block_not = [1]

    def initialize(self, threshold, threshold_low,
                   global_alpha, global_lambda,
                   minibatch_maxsize, device):
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        self.ev.device = device
        self.ev.threshold = threshold
        self.ev.threshold_low = threshold_low
        for gd in self.gd:
            gd.device = device
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
        for forward in self.forward:
            forward.device = device

        # If channels.feed is found - do only forward propagation.
        try:
            # feed = open("/tmp/feed", "rb")
            self.log().info("will open pipe")
            f = os.open("/tmp/feed", os.O_RDONLY)
            self.log().info("pipe opened")
            feed = os.fdopen(f, "rb")
            self.log().info("pipe linked to python descriptor")
            self.switch_to_forward_workflow(feed)
        except FileNotFoundError:
            self.log().info("pipe was not found")
            pass

        retval = self.start_point.initialize_dependent()
        if retval:
            return retval

    def run(self):
        retval = self.start_point.run_dependent()
        if retval:
            return retval
        self.end_point.wait()

    def switch_to_forward_workflow(self, feed):
        self.start_point.unlink()
        self.end_point.unlink()
        self.decision.unlink()
        self.ev.unlink()
        self.loader.unlink()
        self.rpt.unlink()
        for gd in self.gd:
            gd.unlink()
        self.image_saver.unlink()
        self.plt_w.unlink()
        for plt in self.plt:
            plt.unlink()
        for plt_mx in self.plt_mx:
            plt_mx.unlink()
        for forward in self.forward:
            forward.unlink()
        self.rpt.link_from(self.start_point)
        self.loader = UYVYStreamLoader(feed=feed)
        self.loader.link_from(self.rpt)
        self.end_point.link_from(self.loader)
        self.end_point.gate_skip = self.loader.complete
        self.end_point.gate_skip_not = [1]
        self.end_point.gate_block = [0]
        self.end_point.gate_block_not = [0]
        self.forward[0].link_from(self.end_point)
        self.forward[0].input = self.loader.minibatch_data
        for i in range(1, len(self.forward)):
            self.forward[i].link_from(self.forward[i - 1])
        self.plt_result = plotters.ResultPlotter()
        self.plt_result.link_from(self.forward[-1])
        self.plt_result.input = self.forward[-1].max_idx
        self.plt_result.image = self.loader.minibatch_data
        self.rpt.link_from(self.plt_result)


class UYVYStreamLoader(units.Unit):
    """Provides samples from UYVY packed raw video stream.

    Attributes:
        feed: pipe with video stream.
        frame_width: video frame width.
        frame_height: video frame height.
        x: output rectangle left.
        y: output rectangle top.
        width: output rectangle width.
        height: output rectangle height.
        scale: factor to scale frame.
        gray: if grayscale.
    """
    def __init__(self, feed=None, frame_width=1920, frame_height=1080,
                 x=66, y=64, width=464, height=128, scale=0.5, gray=True):
        super(UYVYStreamLoader, self).__init__()
        self.feed = feed
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.scale = scale
        self.gray = gray
        self.minibatch_data = formats.Vector()
        self.minibatch_size = [1]
        self.complete = [0]
        self.cc = None

    def initialize(self):
        self.cy = numpy.zeros([self.height, self.width], dtype=numpy.uint8)
        self.cu = numpy.zeros([self.height, self.width >> 1],
            dtype=numpy.uint8)
        self.cv = numpy.zeros([self.height, self.width >> 1],
            dtype=numpy.uint8)

        self.aw = int(numpy.round(self.width * self.scale))
        self.ah = int(numpy.round(self.height * self.scale))
        if self.gray:
            self.subframe = numpy.zeros([self.ah, self.aw], dtype=numpy.uint8)
        else:
            self.subframe = numpy.zeros([self.ah << 1, self.aw],
                dtype=numpy.uint8)
        self.minibatch_data.v = numpy.zeros([1, self.subframe.shape[0],
            self.subframe.shape[1]], dtype=config.dtypes[config.dtype])

    def run(self):
        if self.complete[0]:
            return
        try:
            n = self.frame_width * self.frame_height * 2
            s = self.feed.read(n)
            frame_img = numpy.frombuffer(s, dtype=numpy.uint8, count=n).\
                reshape(self.frame_height, self.frame_width // 2, 4)
            img = frame_img[self.y:self.y + self.height,
                self.x // 2:(self.x + self.width) // 2]
        except ValueError:
            self.complete[0] = 1
            return
        y = self.cy
        u = self.cu
        v = self.cv

        for row in range(0, img.shape[0]):
            for col in range(0, img.shape[1]):
                pix = img[row, col]
                u[row, col] = pix[0]
                v[row, col] = pix[2]
                y[row, col << 1] = pix[1]
                y[row, (col << 1) + 1] = pix[3]

        if self.scale != 1.0:
            ay = scipy.ndimage.zoom(y, self.scale, order=1)
            if not self.gray:
                au = scipy.ndimage.zoom(u, self.scale, order=1)
                av = scipy.ndimage.zoom(v, self.scale, order=1)
        else:
            ay = y
            au = u
            av = v

        a = self.subframe

        a[:self.ah, :] = ay[:]
        if not self.gray:
            a[self.ah:, :self.aw >> 1] = au
            a[self.ah:, self.aw >> 1:] = av

        sample = self.minibatch_data.v[0]
        sample[:] = a[:]
        normalize(sample)
        self.minibatch_data.update()


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    """
    fin = open("mnist.1.86.2layer100neurons.pickle", "rb")
    w = pickle.load(fin)
    fin.close()

    fout = open("w100.txt", "w")
    weights = w.forward[0].weights.v
    for row in weights:
        fout.write(" ".join("%.6f" % (x) for x in row))
        fout.write("\n")
    fout.close()
    fout = open("b100.txt", "w")
    bias = w.forward[0].bias.v
    fout.write(" ".join("%.6f" % (x) for x in bias))
    fout.write("\n")
    fout.close()

    fout = open("w10.txt", "w")
    weights = w.forward[1].weights.v
    for row in weights:
        fout.write(" ".join("%.6f" % (x) for x in row))
        fout.write("\n")
    fout.close()
    fout = open("b10.txt", "w")
    bias = w.forward[1].bias.v
    fout.write(" ".join("%.6f" % (x) for x in bias))
    fout.write("\n")
    fout.close()

    logging.info("Done")
    sys.exit(0)
    """
    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    try:
        cl = opencl.DeviceList()
        device = cl.get_device()
        try:
            fin = open("%s/cifar.pickle" % (config.snapshot_dir), "rb")
            w = pickle.load(fin)
            fin.close()
        except IOError:
            w = Workflow(layers=[300, 200, 100, 10], device=device)
        w.initialize(threshold=1.0, threshold_low=1.0,
              global_alpha=0.1, global_lambda=0.00005,
              minibatch_maxsize=180, device=device)
    except KeyboardInterrupt:
        return
    try:
        w.run()
    except KeyboardInterrupt:
        w.gd[-1].gate_block = [1]
    """
    logging.info("Will snapshot in 15 seconds...")
    time.sleep(5)
    logging.info("Will snapshot in 10 seconds...")
    time.sleep(5)
    logging.info("Will snapshot in 5 seconds...")
    time.sleep(5)
    fnme = "%s/channels.pickle" % (this_dir)
    logging.info("Snapshotting to %s" % (fnme))
    fout = open(fnme, "wb")
    pickle.dump(w, fout)
    fout.close()
    """

    try:
        plotters.Graphics().wait_finish()
    except:
        pass
    logging.info("End of job")


if __name__ == "__main__":
    main()
