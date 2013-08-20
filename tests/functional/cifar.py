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


import numpy
import units
import formats
import config
import rnd
import opencl
import plotters
import glob
import pickle
import loader
import decision


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
        self.snapshot_time = None
        self.last_snapshot_time = 0
        self.max_idx = None

    def run(self):
        self.input.sync()
        self.output.sync()
        self.max_idx.sync()
        self.indexes.sync()
        self.labels.sync()
        dirnme = self.out_dirs[self.minibatch_class[0]]
        if self.snapshot_time[0] != self.last_snapshot_time:
            self.last_snapshot_time = self.snapshot_time[0]
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
                amp = min(9.0 / 1.7159 / layers[i - 1], 0.05)
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
        self.decision = decision.Decision(snapshot_prefix="cifar")
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err_skipped
        self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        self.image_saver.gate_skip = [0]  # self.decision.just_snapshotted
        self.image_saver.gate_skip_not = [1]
        self.image_saver.snapshot_time = self.decision.snapshot_time

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
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = self.decision.epoch_ended
            self.plt[-1].gate_block_not = [1]
        # Matrix plotter
        # """
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_w = plotters.Weights2D(figure_label="First Layer Weights",
                                        limit=25)
        self.plt_w.input = self.gd[0].weights
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
        retval = self.start_point.initialize_dependent()
        if retval:
            return retval

    def run(self):
        retval = self.start_point.run_dependent()
        if retval:
            return retval
        self.end_point.wait()


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
