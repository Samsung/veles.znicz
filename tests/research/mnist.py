#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

File for MNIST dataset (NN with RELU activation).

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import struct
import os
from zope.interface import implementer

from veles.config import root
import veles.formats as formats
import veles.error as error
import veles.plotting_units as plotting_units
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.conv as conv
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.learning_rate_adjust as lra
import veles.znicz.loader as loader
from veles.znicz.nn_units import NNSnapshotter
from veles.znicz.standard_workflow import StandardWorkflow

mnist_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "samples/MNIST")
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")

root.defaults = {"learning_rate_adjust": {"do": False},
                 "decision": {"fail_iterations": 100},
                 "snapshotter": {"prefix": "mnist"},
                 "loader": {"minibatch_size": 60},
                 "weights_plotter": {"limit": 64},
                 "mnist": {"learning_rate": 0.03,
                           "weights_decay": 0.0,
                           "layers":
                           [{"type": "all2all_tanh", "output_shape": 100,
                             "learning_rate": 0.03, "weights_decay": 0.0,
                             "gradient_moment": 0.9,
                             "weights_filling": "uniform",
                             "bias_filling": "uniform"},
                            {"type": "softmax", "output_shape": 10,
                             "learning_rate": 0.03, "weights_decay": 0.0,
                             "gradient_moment": 0.9,
                             "weights_filling": "uniform",
                             "bias_filling": "uniform"}],
                           "data_paths": {"test_images":  test_image_dir,
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
        fin = open(labels_fnme, "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2049:
            raise error.BadFormatError("Wrong header in train-labels")

        n_labels, = struct.unpack(">i", fin.read(4))
        if n_labels != labels_count:
            raise error.BadFormatError(
                "Wrong number of labels in train-labels")

        arr = numpy.zeros(n_labels, dtype=numpy.byte)
        n = fin.readinto(arr)
        if n != n_labels:
            raise error.BadFormatError("EOF reached while reading labels from "
                                       "train-labels")
        self.original_labels[offs:offs + labels_count] = arr[:]
        if self.original_labels.min() != 0 or self.original_labels.max() != 9:
            raise error.BadFormatError("Wrong labels range in train-labels.")

        fin.close()

        # Reading images:
        fin = open(images_fnme, "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2051:
            raise error.BadFormatError("Wrong header in train-images")

        n_images, = struct.unpack(">i", fin.read(4))
        if n_images != n_labels:
            raise error.BadFormatError(
                "Wrong number of images in train-images")

        n_rows, n_cols = struct.unpack(">2i", fin.read(8))
        if n_rows != 28 or n_cols != 28:
            raise error.BadFormatError("Wrong images size in train-images, "
                                       "should be 28*28")

        # 0 - white, 255 - black
        pixels = numpy.zeros(n_images * n_rows * n_cols, dtype=numpy.ubyte)
        n = fin.readinto(pixels)
        if n != n_images * n_rows * n_cols:
            raise error.BadFormatError("EOF reached while reading images "
                                       "from train-images")

        fin.close()

        # Transforming images into float arrays and normalizing to [-1, 1]:
        images = pixels.astype(numpy.float32).reshape(n_images, n_rows, n_cols)
        self.info("Original range: [%.1f, %.1f]" %
                  (images.min(), images.max()))
        for image in images:
            formats.normalize(image)
        self.info("Range after normalization: [%.1f, %.1f]" %
                  (images.min(), images.max()))
        self.original_data[offs:offs + n_images] = images[:]
        self.info("Done")

    def load_data(self):
        """Here we will load MNIST data.
        """
        self.original_labels = numpy.zeros([70000], dtype=numpy.int8)
        self.original_data = numpy.zeros([70000, 28, 28], dtype=numpy.float32)

        self.load_original(0, 10000,
                           root.mnist.data_paths.test_label,
                           root.mnist.data_paths.test_images)
        self.load_original(10000, 60000,
                           root.mnist.data_paths.train_label,
                           root.mnist.data_paths.train_images)

        self.class_lengths[0] = 0
        self.class_lengths[1] = 10000
        self.class_lengths[2] = 60000


class Workflow(StandardWorkflow):
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
                             minibatch_size=root.loader.minibatch_size)
        self.loader.link_from(self.repeater)

        # Add fwds units
        self.parse_forwards_from_config()

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.fwds[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("labels", "minibatch_labels"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.DecisionGD(
            self, fail_iterations=root.decision.fail_iterations)
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
        self.decision.fwds = self.fwds
        self.decision.gds = self.gds
        self.decision.evaluator = self.evaluator

        self.info("root.snapshotter.prefix %s" % root.snapshotter.prefix)
        self.info("root.common.snapshot_dir %s" % root.common.snapshot_dir)
        self.snapshotter = NNSnapshotter(self, prefix=root.snapshotter.prefix,
                                         directory=root.common.snapshot_dir)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.decision.epoch_ended | ~self.decision.improved)

        # Add gradient descent units
        self.create_gradient_descent_units()

        if root.learning_rate_adjust.do:
            # Add learning_rate_adjust unit

            self.learning_rate_adjust = lra.LearningRateAdjust(
                self, lr_function=lra.inv_adjust_policy(
                    0.01, 0.0001, 0.75))
            self.learning_rate_adjust.link_from(self.gds[0])
            self.learning_rate_adjust.add_gd_units(self.gds)

            self.repeater.link_from(self.learning_rate_adjust)
            self.end_point.link_from(self.learning_rate_adjust)
        else:
            self.repeater.link_from(self.gds[0])
            self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Confusion matrix plotter
        """
        self.plt_mx = []
        for i in range(1, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].link_attrs(self.decision,
                                       ("input", "confusion_matrixes"))
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended
        """
        # err_y plotter
        self.plt_err_y = []
        for i in range(1, 3):
            self.plt_err_y.append(plotting_units.AccumulatingPlotter(
                self, name="Last layer max gradient sum",
                plot_style=styles[i]))
            self.plt_err_y[-1].link_attrs(self.decision,
                                          ("input", "max_err_y_sums"))
            self.plt_err_y[-1].input_field = i
            self.plt_err_y[-1].link_from(self.decision)
            self.plt_err_y[-1].gate_block = ~self.decision.epoch_ended
        self.plt_err_y[0].clear_plot = True
        self.plt_err_y[-1].redraw_plot = True

        # Weights plotter
        self.plt_mx = []
        prev_channels = 1
        for i in range(len(layers)):
            if (not isinstance(self.fwds[i], conv.Conv) and
                    not isinstance(self.fwds[i], all2all.All2All)):
                continue
            self.decision.vectors_to_sync[self.gds[i].gradient_weights] = 1
            nme = "%s %s" % (i + 1, layers[i]["type"])
            print("Added:", nme)
            plt_mx = nn_plotting_units.Weights2D(
                self, name=nme, limit=root.weights_plotter.limit)
            self.plt_mx.append(plt_mx)
            self.plt_mx[-1].link_attrs(self.gds[i], ("input",
                                                     "gradient_weights"))
            self.plt_mx[-1].input_field = "mem"
            if isinstance(self.fwds[i], conv.Conv):
                self.plt_mx[-1].get_shape_from = (
                    [self.fwds[i].kx, self.fwds[i].ky, prev_channels])
                prev_channels = self.fwds[i].n_kernels
            if (layers[i].get("output_shape") is not None and
                    layers[i]["type"] != "softmax"):
                self.plt_mx[-1].link_attrs(self.fwds[i],
                                           ("get_shape_from", "input"))
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended

    def initialize(self, learning_rate, weights_decay, device):
        super(Workflow, self).initialize(learning_rate=learning_rate,
                                         weights_decay=weights_decay,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.mnist.layers)
    main(learning_rate=root.mnist.learning_rate,
         weights_decay=root.mnist.weights_decay)
