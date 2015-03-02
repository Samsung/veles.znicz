#!/usr/bin/python3 -O
"""
Created on August 12, 2013

Model created for digits recognition. Database - MNIST. Model - fully-connected
Neural Network with MSE loss function with target encoded as ideal image
(784 points).

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import logging
import os
import sys

import numpy
import six
from zope.interface import implementer

from veles.external.freetype import (Face,  # pylint: disable=E0611
                                     FT_Matrix, FT_LOAD_RENDER, FT_Vector,
                                     FT_Set_Transform, byref)
from veles.config import root
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.loader as loader
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.nn_units import NNSnapshotter


sys.path.append(os.path.dirname(__file__))
from .loader_mnist import MnistLoader


root.mnist784.update({
    "decision": {"fail_iterations": 100, "max_epochs": 100000},
    "snapshotter": {"prefix": "mnist_784"},
    "loader": {"minibatch_size": 100, "force_cpu": False},
    "weights_plotter": {"limit": 16},
    "learning_rate": 0.00001,
    "weights_decay": 0.00005,
    "layers": [784, 784],
    "data_paths": {"arial":
                   os.path.join(root.common.test_dataset_root, "arial.ttf")}})


def do_plot(fontPath, text, size, angle, sx, sy,
            randomizePosition, SX, SY):
    if six.PY2:
        face = Face(six.binary_type(fontPath))
    else:
        face = Face(six.binary_type(fontPath, 'UTF-8'))
    # face.set_char_size(48 * 64)
    face.set_pixel_sizes(0, size)

    c = text[0]

    angle = (angle / 180.0) * numpy.pi

    mx_r = numpy.array([[numpy.cos(angle), -numpy.sin(angle)],
                        [numpy.sin(angle), numpy.cos(angle)]],
                       dtype=numpy.double)
    mx_s = numpy.array([[sx, 0.0],
                        [0.0, sy]], dtype=numpy.double)

    mx = numpy.dot(mx_s, mx_r)

    matrix = FT_Matrix((int)(mx[0, 0] * 0x10000),
                       (int)(mx[0, 1] * 0x10000),
                       (int)(mx[1, 0] * 0x10000),
                       (int)(mx[1, 1] * 0x10000))
    flags = FT_LOAD_RENDER
    pen = FT_Vector(0, 0)
    FT_Set_Transform(face._FT_Face, byref(matrix), byref(pen))

    j = 0
    while True:
        slot = face.glyph
        if not face.get_char_index(c):
            return None
        face.load_char(c, flags)
        bitmap = slot.bitmap
        width = bitmap.width
        height = bitmap.rows
        if width > SX or height > SY:
            j = j + 1
            face.set_pixel_sizes(0, size - j)
            continue
        break

    if randomizePosition:
        x = int(numpy.floor(numpy.random.rand() * (SX - width)))
        y = int(numpy.floor(numpy.random.rand() * (SY - height)))
    else:
        x = int(numpy.floor((SX - width) * 0.5))
        y = int(numpy.floor((SY - height) * 0.5))

    img = numpy.zeros([SY, SX], dtype=numpy.uint8)
    img[y:y + height, x: x + width] = numpy.array(
        bitmap.buffer, dtype=numpy.uint8).reshape(height, width)
    if img.max() == img.min():
        logging.info("Font %s returned empty glyph" % (fontPath))
        return None
    return img


@implementer(loader.IFullBatchLoader)
class Mnist784Loader(MnistLoader, loader.FullBatchLoaderMSE):
    """Loads MNIST dataset.
    """
    def load_data(self):
        """Here we will load MNIST data.
        """
        super(Mnist784Loader, self).load_data()
        self.class_targets.reset()
        self.class_targets.mem = numpy.zeros(
            [10, 784], dtype=opencl_types.dtypes[root.common.precision_type])
        for i in range(0, 10):
            img = do_plot(root.mnist784.data_paths.arial,
                          "%d" % (i,), 28, 0.0, 1.0, 1.0, False, 28, 28)
            self.class_targets[i] = img.ravel().astype(
                opencl_types.dtypes[root.common.precision_type])
        # normalization
        self.original_targets.mem = numpy.zeros(
            [len(self.original_labels), self.class_targets.mem.shape[1]],
            dtype=self.original_data.dtype)
        for i, label in enumerate(self.original_labels):
            self.original_targets[i] = self.class_targets[label]


class Mnist784Workflow(nn_units.NNWorkflow):
    """
    Model created for digits recognition. Database - MNIST. Model -
    fully-connected Neural Network with MSE loss function with target encoded
    as ideal image (784 points).
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        kwargs["layers"] = layers
        super(Mnist784Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Mnist784Loader(
            self, **root.mnist784.loader.__content__)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.forwards[:]
        for i in range(0, len(layers)):
            aa = all2all.All2AllTanh(self, output_sample_shape=[layers[i]])
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
        self.evaluator = evaluator.EvaluatorMSE(self)
        self.evaluator.link_from(self.forwards[-1])
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("target", "minibatch_targets"),
                                  ("labels", "minibatch_labels"),
                                  "class_targets")
        self.evaluator.link_attrs(self.forwards[-1], "output")

        # Add decision unit
        self.decision = decision.DecisionMSE(
            self, fail_iterations=root.mnist784.decision.fail_iterations,
            max_epochs=root.mnist784.decision.max_epochs)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class", "minibatch_size",
                                 "last_minibatch", "class_lengths",
                                 "epoch_ended", "epoch_number",
                                 "minibatch_offset", "minibatch_size")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_metrics", "metrics"),
            ("minibatch_mse", "mse"))

        self.snapshotter = NNSnapshotter(
            self, prefix=root.mnist784.snapshotter.prefix,
            directory=root.common.snapshot_dir)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.decision.epoch_ended | ~self.decision.improved)

        # Add ImagePlotter Saver unit
        self.image_saver = image_saver.ImageSaver(self)
        self.image_saver.link_from(self.snapshotter)
        self.image_saver.link_attrs(self.evaluator, "output", "target")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indices", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")
        self.image_saver.gate_skip = ~self.decision.improved
        self.image_saver.link_attrs(self.snapshotter,
                                    ("this_save_time", "time"))

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(None for i in range(0, len(self.forwards)))
        self.gds[-1] = gd.GDTanh(self)
        self.gds[-1].link_from(self.image_saver)
        self.gds[-1].link_attrs(self.forwards[-1], "output", "input",
                                "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.forwards) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.forwards[i], "output", "input",
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader, ("batch_size",
                                                 "minibatch_size"))
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_metrics"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else
                                   self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True

        # Weights plotter
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="First Layer Weights",
            limit=root.mnist784.weights_plotter.limit)
        self.plt_mx.link_attrs(self.gds[0], ("input", "weights"))
        self.plt_mx.input_field = "mem"
        self.plt_mx.link_attrs(self.forwards[0], ("get_shape_from", "input"))
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended

        # Image plotter
        self.plt_img = plotting_units.ImagePlotter(self, name="output sample")
        self.plt_img.inputs.append(self.forwards[-1].output)
        self.plt_img.input_fields.append(0)
        self.plt_img.inputs.append(self.forwards[0].input)
        self.plt_img.input_fields.append(0)
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_skip = ~self.decision.epoch_ended

        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(0, 3):
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_max[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offset = 1
            self.plt_max[-1].link_from(self.plt[-1] if not i else
                                       self.plt_max[-2])
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(0, 3):
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offset = 2
            self.plt_min[-1].link_from(self.plt_max[-1] if not i else
                                       self.plt_min[-2])
        self.plt_min[-1].redraw_plot = True

    def initialize(self, learning_rate, weights_decay, device, **kwargs):
        super(Mnist784Workflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            device=device)


def run(load, main):
    load(Mnist784Workflow, layers=root.mnist784.layers)
    main(learning_rate=root.mnist784.learning_rate,
         weights_decay=root.mnist784.weights_decay)
