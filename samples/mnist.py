#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

File for MNIST dataset.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import struct

from veles.config import root
import veles.error as error
import veles.formats as formats
import veles.plotting_units as plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader
from veles.interaction import Shell
from veles.external.progressbar import ProgressBar


test_image_dir = os.path.join(
    root.common.veles_dir, "veles/znicz/samples/MNIST/t10k-images.idx3-ubyte")
test_label_dir = os.path.join(
    root.common.veles_dir, "veles/znicz/samples/MNIST/t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(
    root.common.veles_dir, "veles/znicz/samples/MNIST/train-images.idx3-ubyte")
train_label_dir = os.path.join(
    root.common.veles_dir, "veles/znicz/samples/MNIST/train-labels.idx1-ubyte")


root.defaults = {"all2all": {"weights_magnitude": 0.05},
                 "decision": {"fail_iterations": 100,
                              "snapshot_prefix": "mnist",
                              "store_samples_mse": True},
                 "loader": {"minibatch_maxsize": 60},
                 "mnist": {"global_alpha": 0.01,
                           "global_lambda": 0.0,
                           "layers": [100, 10],
                           "path_for_load_data": {"test_images":
                                                  test_image_dir,
                                                  "test_label":
                                                  test_label_dir,
                                                  "train_images":
                                                  train_image_dir,
                                                  "train_label":
                                                  train_label_dir}}}


class Loader(loader.FullBatchLoader):
    """Loads MNIST dataset.
    """
    def load_original(self, offs, labels_count, labels_fnme, images_fnme):
        """Loads data from original MNIST files.
        """
        self.info("Loading from original MNIST files...")

        # Reading labels:
        fin = open(labels_fnme, "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2049:
            raise error.ErrBadFormat("Wrong header in train-labels")

        n_labels, = struct.unpack(">i", fin.read(4))
        if n_labels != labels_count:
            raise error.ErrBadFormat("Wrong number of labels in train-labels")

        arr = numpy.zeros(n_labels, dtype=numpy.byte)
        n = fin.readinto(arr)
        if n != n_labels:
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
        pixels = numpy.zeros(n_images * n_rows * n_cols, dtype=numpy.ubyte)
        n = fin.readinto(pixels)
        if n != n_images * n_rows * n_cols:
            raise error.ErrBadFormat("EOF reached while reading images "
                                     "from train-images")

        fin.close()

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
        self.original_labels = numpy.zeros([70000], dtype=numpy.int8)
        self.original_data = numpy.zeros([70000, 28, 28], dtype=numpy.float32)

        self.load_original(0, 10000, root.mnist.path_for_load_data.test_label,
                           root.mnist.path_for_load_data.test_images)
        self.load_original(10000, 60000,
                           root.mnist.path_for_load_data.train_label,
                           root.mnist.path_for_load_data.train_images)

        self.class_samples[0] = 0
        self.class_samples[1] = 10000
        self.class_samples[2] = 60000


class Workflow(nn_units.NNWorkflow):
    """Workflow for MNIST dataset (handwritten digits recognition).
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        kwargs["name"] = kwargs.get("name", "MNIST")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, name="Mnist fullbatch loader",
                             minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.fwds[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                         device=device)
            else:
                aa = all2all.All2AllSoftmax(self,
                                            output_shape=[layers[i]],
                                            device=device)
            self.fwds.append(aa)
            if i:
                self.fwds[i].link_from(self.fwds[i - 1])
                self.fwds[i].link_attrs(self.fwds[i - 1],
                                        ("input", "output"))
            else:
                self.fwds[i].link_from(self.loader)
                self.fwds[i].link_attrs(self.loader,
                                        ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.fwds[-1], ("y", "output"), "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("labels", "minibatch_labels"),
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.Decision(self)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "no_more_minibatches_left",
                                 "class_samples")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_y_sum"))

        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.decision.epoch_ended

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(list(None for i in range(0, len(self.fwds))))
        self.gds[-1] = gd.GDSM(self, device=device)
        self.gds[-1].link_from(self.ipython)
        self.gds[-1].link_attrs(self.evaluator, "err_y")
        self.gds[-1].link_attrs(self.fwds[-1],
                                ("y", "output"),
                                ("h", "input"),
                                "weights", "bias")
        self.gds[-1].gate_skip = self.decision.gd_skip
        self.gds[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.gds[i + 1], ("err_y", "err_h"))
            self.gds[i].link_attrs(self.fwds[i],
                                   ("y", "output"),
                                   ("h", "input"),
                                   "weights", "bias")
            self.gds[i].gate_skip = self.decision.gd_skip
            self.gds[i].link_attrs(self.loader,
                                   ("batch_size", "minibatch_size"))
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["g-", "r-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="Errors", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if i == 1 else self.plt[-2])
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
        for i in range(1, 3):
            self.plt_err_y.append(plotting_units.AccumulatingPlotter(
                self, name="Last layer max gradient sum",
                fit_poly_power=3, plot_style=styles[i]))
            self.plt_err_y[-1].link_attrs(self.decision,
                                          ("input", "max_err_y_sums"))
            self.plt_err_y[-1].input_field = i
            self.plt_err_y[-1].link_from(self.decision)
            self.plt_err_y[-1].gate_block = ~self.decision.epoch_ended
        self.plt_err_y[0].clear_plot = True
        self.plt_err_y[-1].redraw_plot = True

    def initialize(self, global_alpha, global_lambda, device):
        return super(Workflow, self).initialize(global_alpha=global_alpha,
                                                global_lambda=global_lambda,
                                                device=device)


def run(load, main):
    load(Workflow, layers=root.mnist.layers)
    main(global_alpha=root.mnist.global_alpha,
         global_lambda=root.mnist.global_lambda)
