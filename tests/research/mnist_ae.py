#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

File for MNIST dataset.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import struct
from zope.interface import implementer

from veles.config import root
import veles.error as error
import veles.formats as formats
import veles.plotting_units as plotting_units
from veles.znicz.nn_units import NNSnapshotter
import veles.znicz.nn_units as nn_units
import veles.znicz.nn_plotting_units as nn_plotting_units
#import veles.znicz.all2all as all2all
import veles.znicz.conv as conv
import veles.znicz.deconv as deconv
import veles.znicz.gd_deconv as gd_deconv
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.pooling as pooling
import veles.znicz.gd_pooling as gd_pooling
#import veles.znicz.gd as gd
#import veles.znicz.gd_conv as gd_conv
import veles.znicz.loader as loader
#from veles.interaction import Shell
from veles.external.progressbar import ProgressBar
#from veles.mutable import Bool


mnist_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "samples/MNIST")
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")


root.defaults = {"all2all": {"weights_stddev": 0.05},
                 "decision": {"fail_iterations": 20,
                              "store_samples_mse": True},
                 "snapshotter": {"prefix": "mnist"},
                 "loader": {"minibatch_size": 100},
                 "mnist": {"learning_rate": 0.03,
                           "weights_decay": 0.0,
                           "layers": [100, 10],
                           "data_paths": {"test_images": test_image_dir,
                                          "test_label": test_label_dir,
                                          "train_images": train_image_dir,
                                          "train_label": train_label_dir}}}


@implementer(loader.IFullBatchLoader)
class Loader(loader.FullBatchLoader):
    """Loads MNIST dataset.
    """
    def load_original(self, offs, labels_count, labels_fnme, images_fnme):
        """Loads data from original MNIST files.
        """
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
            self.original_labels[offs:offs + labels_count] = arr[:]
            if (self.original_labels.min() != 0 or
                    self.original_labels.max() != 9):
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
        self.original_data[offs:offs + n_images] = images[:]
        self.info("Range after normalization: [%.1f, %.1f]" %
                  (images.min(), images.max()))

    def load_data(self):
        """Here we will load MNIST data.
        """
        if hasattr(self, "_inited"):
            raise ValueError("Failed")
        self._inited = True
        self.original_labels = numpy.zeros([70000], dtype=numpy.int8)
        self.original_data = numpy.zeros([70000, 28, 28], dtype=numpy.float32)

        self.load_original(0, 10000, root.mnist.data_paths.test_label,
                           root.mnist.data_paths.test_images)
        self.load_original(10000, 60000,
                           root.mnist.data_paths.train_label,
                           root.mnist.data_paths.train_images)

        self.class_lengths[0] = 0
        self.class_lengths[1] = 10000
        self.class_lengths[2] = 60000

        self.original_data.shape = [70000, 28, 28, 1]


class Workflow(nn_units.NNWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    """
    def __init__(self, workflow, layers, **kwargs):
        kwargs["name"] = kwargs.get("name", "MNIST")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, name="MNIST loader",
                             minibatch_size=root.loader.minibatch_size)
        self.loader.link_from(self.repeater)

        LR = 0.000001
        WD = 0.00005
        GM = 0.00001
        KX = 5
        KY = 5
        N_KERNELS = 5

        unit = conv.Conv(self, n_kernels=N_KERNELS, kx=KX, ky=KY,
                         weights_filling="uniform", include_bias=False)
        unit.link_from(self.loader)
        unit.link_attrs(self.loader, ("input", "minibatch_data"))
        self.conv = unit

        unit = pooling.StochasticAbsPooling(self, kx=3, ky=3, sliding=(2, 2))
        unit.link_from(self.conv)
        unit.link_attrs(self.conv, ("input", "output"))
        self.pool = unit

        unit = gd_pooling.GDMaxAbsPooling(
            self, kx=self.pool.kx, ky=self.pool.ky, sliding=(2, 2))
        unit.link_from(self.pool)
        unit.link_attrs(self.pool, "input", "input_offset",
                        ("err_output", "output"))
        self.depool = unit

        unit = deconv.Deconv(
            self, n_kernels=N_KERNELS, kx=KX, ky=KY,
            sliding=self.conv.sliding, padding=self.conv.padding)
        self.deconv = unit
        unit.link_from(self.depool)
        unit.link_attrs(self.conv, "weights")
        unit.link_attrs(self.depool, ("input", "err_input"))

        # Add evaluator for single minibatch
        unit = evaluator.EvaluatorMSE(self)
        self.evaluator = unit
        unit.link_from(self.deconv)
        unit.link_attrs(self.deconv, "output")
        unit.link_attrs(self.loader,
                        ("batch_size", "minibatch_size"),
                        ("target", "minibatch_data"))

        # Add decision unit
        unit = decision.DecisionMSE(
            self, fail_iterations=root.decision.fail_iterations)
        self.decision = unit
        unit.link_from(self.evaluator)
        unit.link_attrs(self.loader, "minibatch_class",
                        "minibatch_size", "last_minibatch",
                        "class_lengths", "epoch_ended",
                        "epoch_number")
        unit.link_attrs(self.evaluator, ("minibatch_metrics", "metrics"))

        unit = NNSnapshotter(self, prefix=root.snapshotter.prefix,
                             directory=root.common.snapshot_dir,
                             compress="", time_interval=0)
        self.snapshotter = unit
        unit.link_from(self.decision)
        unit.link_attrs(self.decision, ("suffix", "snapshot_suffix"))
        unit.gate_skip = ~self.loader.epoch_ended | ~self.decision.improved

        self.end_point.link_from(self.snapshotter)
        self.end_point.gate_block = ~self.decision.complete

        # Add gradient descent units
        unit = gd_deconv.GDDeconv(
            self, n_kernels=N_KERNELS, kx=KX, ky=KY,
            sliding=self.conv.sliding, padding=self.conv.padding,
            learning_rate=LR, weights_decay=WD, gradient_moment=GM)
        self.gd_deconv = unit
        unit.link_attrs(self.evaluator, "err_output")
        unit.link_attrs(self.deconv, "weights", "input")
        unit.gate_skip = self.decision.gd_skip

        self.gd_deconv.need_err_input = False
        self.repeater.link_from(self.gd_deconv)

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        prev = self.snapshotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(prev)
            self.plt[-1].gate_skip = ~self.decision.epoch_ended
            prev = self.plt[-1]
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True

        # Weights plotter
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="Weights", limit=64)
        self.plt_mx.link_attrs(self.conv, ("input", "weights"))
        self.plt_mx.get_shape_from = [KX, KY, 1]
        self.plt_mx.input_field = "mem"
        self.plt_mx.link_from(prev)
        self.plt_mx.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_mx

        # Input plotter
        """
        self.plt_inp = nn_plotting_units.Weights2D(
            self, name="First Layer Input", limit=64)
        self.plt_inp.link_attrs(self.conv, "input")
        self.plt_inp.get_shape_from = [28, 28, 1]
        self.plt_inp.input_field = "mem"
        self.plt_inp.link_from(prev)
        self.plt_inp.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_inp
        """

        # Output plotter
        """
        self.plt_out = nn_plotting_units.Weights2D(
            self, name="First Layer Output", limit=64)
        self.plt_out.link_attrs(self.conv, ("input", "output"))
        self.plt_out.get_shape_from = [28 - KX + 1, 28 - KY + 1, N_KERNELS]
        self.plt_out.input_field = "mem"
        self.plt_out.link_from(prev)
        self.plt_out.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_out
        """

        # Deconv result plotter
        self.plt_out = nn_plotting_units.Weights2D(
            self, name="Deconv result", limit=64)
        self.plt_out.link_attrs(self.deconv, ("input", "output"))
        self.plt_out.get_shape_from = [28, 28, 1]
        self.plt_out.input_field = "mem"
        self.plt_out.link_from(prev)
        self.plt_out.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_out

        self.gd_deconv.link_from(prev)
        self.gd_deconv.gate_block = self.decision.complete


def run(load, main):
    load(Workflow, layers=root.mnist.layers)
    main()
