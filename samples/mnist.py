#!/usr/bin/python3 -O
"""
Created on Mar 20, 2013

Model created for digits recognition. Database - MNIST.
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import struct
from zope.interface import implementer

from veles.config import root, get
import veles.error as error
import veles.formats as formats
import veles.plotting_units as plotting_units
from veles.znicz.nn_units import NNSnapshotter
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader
from veles.interaction import Shell
from veles.external.progressbar import ProgressBar


mnist_dir = os.path.abspath(get(
    root.mnist.loader.base, os.path.join(os.path.dirname(__file__), "MNIST")))
if not os.access(mnist_dir, os.W_OK):
    # Fall back to ~/.veles/MNIST
    mnist_dir = os.path.join(root.common.veles_user_dir, "MNIST")
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")


root.mnist.update({
    "all2all": {"weights_stddev": 0.05},
    "decision": {"fail_iterations": 100,
                 "max_epochs": 1000000000},
    "snapshotter": {"prefix": "mnist"},
    "loader": {"minibatch_size": 60, "on_device": True},
    "learning_rate": 0.03,
    "weights_decay": 0.0005,  # 1.6%
    "factor_ortho": 0.0,
    "layers": [100, 10]})


@implementer(loader.IFullBatchLoader)
class MnistLoader(loader.FullBatchLoader):
    """Loads MNIST dataset.
    """
    def load_original(self, offs, labels_count, labels_fnme, images_fnme):
        """Loads data from original MNIST files.
        """
        if not os.path.exists(mnist_dir):
            url = "http://yann.lecun.com/exdb/mnist"
            self.warning("%s does not exist, downloading from %s...",
                         mnist_dir, url)

            import gzip
            import wget

            files = {"train-images-idx3-ubyte.gz": "train-images.idx3-ubyte",
                     "train-labels-idx1-ubyte.gz": "train-labels.idx1-ubyte",
                     "t10k-images-idx3-ubyte.gz": "t10k-images.idx3-ubyte",
                     "t10k-labels-idx1-ubyte.gz": "t10k-labels.idx1-ubyte"}

            os.mkdir(mnist_dir)
            for index, (k, v) in enumerate(sorted(files.items())):
                self.info("%d/%d", index + 1, len(files))
                wget.download("%s/%s" % (url, k), mnist_dir)
                print("")
                with open(os.path.join(mnist_dir, v), "wb") as fout:
                    gz_file = os.path.join(mnist_dir, k)
                    with gzip.GzipFile(gz_file) as fin:
                        fout.write(fin.read())
                    os.remove(gz_file)

        self.info("Loading from original MNIST files...")

        # Reading labels:
        with open(labels_fnme, "rb") as fin:
            header, = struct.unpack(">i", fin.read(4))
            if header != 2049:
                raise error.BadFormatError("Wrong header in train-labels")

            n_labels, = struct.unpack(">i", fin.read(4))
            if n_labels != labels_count:
                raise error.BadFormatError("Wrong number of labels in "
                                           "train-labels")

            arr = numpy.zeros(n_labels, dtype=numpy.byte)
            n = fin.readinto(arr)
            if n != n_labels:
                raise error.BadFormatError("EOF reached while reading labels "
                                           "from train-labels")
            self.original_labels.mem[offs:offs + labels_count] = arr[:]
            if (self.original_labels.mem.min() != 0 or
                    self.original_labels.mem.max() != 9):
                raise error.BadFormatError(
                    "Wrong labels range in train-labels.")

        # Reading images:
        with open(images_fnme, "rb") as fin:
            header, = struct.unpack(">i", fin.read(4))
            if header != 2051:
                raise error.BadFormatError("Wrong header in train-images")

            n_images, = struct.unpack(">i", fin.read(4))
            if n_images != n_labels:
                raise error.BadFormatError("Wrong number of images in "
                                           "train-images")

            n_rows, n_cols = struct.unpack(">2i", fin.read(8))
            if n_rows != 28 or n_cols != 28:
                raise error.BadFormatError("Wrong images size in train-images,"
                                           " should be 28*28")

            # 0 - white, 255 - black
            pixels = numpy.zeros(n_images * n_rows * n_cols, dtype=numpy.ubyte)
            n = fin.readinto(pixels)
            if n != n_images * n_rows * n_cols:
                raise error.BadFormatError("EOF reached while reading images "
                                           "from train-images")

        # Transforming images into float arrays and normalizing to [-1, 1]:
        images = pixels.astype(numpy.float32).reshape(n_images, n_rows, n_cols)
        self.info("Original range: [%.1f, %.1f];"
                  " performing normalization..." % (images.min(),
                                                    images.max()))
        progress = ProgressBar(maxval=len(images), term_width=17)
        progress.start()
        for image in images:
            progress.inc()
            formats.normalize(image)
        progress.finish()
        self.original_data.mem[offs:offs + n_images] = images[:]
        self.info("Range after normalization: [%.1f, %.1f]" %
                  (images.min(), images.max()))

    def load_data(self):
        """Here we will load MNIST data.
        """
        self.original_labels.mem = numpy.zeros([70000], dtype=numpy.int32)
        self.original_data.mem = numpy.zeros([70000, 28, 28],
                                             dtype=numpy.float32)

        self.load_original(0, 10000, test_label_dir, test_image_dir)
        self.load_original(10000, 60000, train_label_dir, train_image_dir)

        self.class_lengths[0] = 0
        self.class_lengths[1] = 10000
        self.class_lengths[2] = 60000


class MnistWorkflow(nn_units.NNWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    """
    def __init__(self, workflow, layers, **kwargs):
        kwargs["name"] = kwargs.get("name", "MNIST")
        super(MnistWorkflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = MnistLoader(
            self, name="Mnist fullbatch loader",
            minibatch_size=root.mnist.loader.minibatch_size,
            on_device=root.mnist.loader.on_device)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.forwards[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(
                    self, output_shape=[layers[i]],
                    weights_stddev=root.mnist.all2all.weights_stddev)
            else:
                aa = all2all.All2AllSoftmax(
                    self, output_shape=[layers[i]],
                    weights_stddev=root.mnist.all2all.weights_stddev)
            self.forwards.append(aa)
            if i:
                self.forwards[i].link_from(self.forwards[i - 1])
                self.forwards[i].link_attrs(
                    self.forwards[i - 1], ("input", "output"))
            else:
                self.forwards[i].link_from(self.loader)
                self.forwards[i].link_attrs(
                    self.loader, ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self)
        self.evaluator.link_from(self.forwards[-1])
        self.evaluator.link_attrs(self.forwards[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("labels", "minibatch_labels"),
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.DecisionGD(
            self, fail_iterations=root.mnist.decision.fail_iterations,
            max_epochs=root.mnist.decision.max_epochs)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class", "minibatch_size",
                                 "last_minibatch", "class_lengths",
                                 "epoch_ended", "epoch_number")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_output_sum"))

        self.snapshotter = NNSnapshotter(
            self, prefix=root.mnist.snapshotter.prefix,
            directory=root.common.snapshot_dir, time_interval=0)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.loader.epoch_ended | ~self.decision.improved)

        self.ipython = Shell(self)
        self.ipython.link_from(self.snapshotter)
        self.ipython.gate_skip = ~self.decision.epoch_ended

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(list(None for i in range(0, len(self.forwards))))
        self.gds[-1] = gd.GDSM(self,
                               learning_rate=root.mnist.learning_rate)
        self.gds[-1].link_from(self.ipython)
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.forwards[-1],
                                ("output", "output"),
                                ("input", "input"),
                                "weights", "bias")
        self.gds[-1].gate_skip = self.decision.gd_skip
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        for i in range(len(self.forwards) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(
                self, learning_rate=root.mnist.learning_rate,
                factor_ortho=root.mnist.factor_ortho)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].link_attrs(self.forwards[i], "output", "input",
                                   "weights", "bias")
            self.gds[i].gate_skip = self.decision.gd_skip
            self.gds[i].link_attrs(self.loader,
                                   ("batch_size", "minibatch_size"))
        self.gds[0].need_err_input = False
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        self.slaves_plotter = plotting_units.SlaveStats(self)
        self.slaves_plotter.link_from(self.decision)

        # Error plotter
        self.plt = []
        styles = ["g-", "r-", "k-"]
        for i, style in enumerate(styles):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="Errors", plot_style=style))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i + 1
            if i == 0:
                self.plt[-1].link_from(self.decision)
            else:
                self.plt[-1].link_from(self.plt[-2])
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True

        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(1, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].link_attrs(self.decision,
                                       ("input", "confusion_matrixes"))
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended

        # err_y plotter
        self.plt_err_y = []
        for i, style in enumerate(styles):
            self.plt_err_y.append(plotting_units.AccumulatingPlotter(
                self, name="Last layer max gradient sum",
                fit_poly_power=3, plot_style=style))
            self.plt_err_y[-1].link_attrs(self.decision,
                                          ("input", "max_err_y_sums"))
            self.plt_err_y[-1].input_field = i + 1
            if i == 0:
                self.plt_err_y[-1].link_from(self.decision)
            else:
                self.plt_err_y[-1].link_from(self.plt_err_y[-2])
            self.plt_err_y[-1].gate_block = ~self.decision.epoch_ended
        self.plt_err_y[0].clear_plot = True
        self.plt_err_y[-1].redraw_plot = True

    def initialize(self, learning_rate, weights_decay, device, **kwargs):
        return super(MnistWorkflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            device=device)


def run(load, main):
    load(MnistWorkflow, layers=root.mnist.layers)
    main(learning_rate=root.mnist.learning_rate,
         weights_decay=root.mnist.weights_decay)
