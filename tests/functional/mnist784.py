#!/usr/bin/python3.3 -O
"""
Created on August 12, 2013

MNIST with target encoded as ideal image (784 points), MSE.

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
add_path("%s" % (this_dir))
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import numpy
import config
import znicz_config
import rnd
import opencl
import plotters
import logging
import mnist
import formats
import decision
import image_saver
import all2all
import evaluator
import gd


add_path("/usr/local/lib/python3.3/dist-packages/freetype")
add_path("/usr/local/lib/python3.3/dist-packages/freetype/ft_enums")
from freetype import *


def do_plot(fontPath, text, size, angle, sx, sy,
            randomizePosition, SX, SY):
    face = Face(bytes(fontPath, 'UTF-8'))
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
            # logging.info("Set pixel size for font %s to %d" % (
            #    fontPath, size - j))
            continue
        break

    if randomizePosition:
        x = int(numpy.floor(numpy.random.rand() * (SX - width)))
        y = int(numpy.floor(numpy.random.rand() * (SY - height)))
    else:
        x = int(numpy.floor((SX - width) * 0.5))
        y = int(numpy.floor((SY - height) * 0.5))

    img = numpy.zeros([SY, SX], dtype=numpy.uint8)
    img[y:y + height, x: x + width] = numpy.array(bitmap.buffer,
        dtype=numpy.uint8).reshape(height, width)
    if img.max() == img.min():
        logging.info("Font %s returned empty glyph" % (fontPath))
        return None
    return img


class Loader(mnist.Loader):
    """Loads MNIST dataset.
    """
    def load_data(self):
        """Here we will load MNIST data.
        """
        super(Loader, self).load_data()
        self.class_target.reset()
        self.class_target.v = numpy.zeros([10, 784],
            dtype=config.dtypes[config.dtype])
        for i in range(0, 10):
            img = do_plot("%s/arial.ttf" % (config.test_dataset_root,),
                          "%d" % (i,), 28, 0.0, 1.0, 1.0, False, 28, 28)
            self.class_target.v[i] = img.ravel().astype(
                                config.dtypes[config.dtype])
            formats.normalize(self.class_target.v[i])
        self.original_target = numpy.zeros([self.original_labels.shape[0],
                                            self.class_target.v.shape[1]],
            dtype=config.dtypes[config.dtype])
        for i in range(0, self.original_labels.shape[0]):
            label = self.original_labels[i]
            self.original_target[i] = self.class_target.v[label]


import workflow


class Workflow(workflow.OpenCLWorkflow):
    """Sample workflow.
    """
    def __init__(self, layers=None, device=None):
        super(Workflow, self).__init__(device=device)

        self.rpt.link_from(self.start_point)

        self.loader = Loader()
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward.clear()
        for i in range(0, len(layers)):
            aa = all2all.All2AllTanh([layers[i]], device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorMSE(device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.target = self.loader.minibatch_target
        self.ev.class_target = self.loader.class_target
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision()
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver()
        self.image_saver.link_from(self.decision)
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.ev.y
        self.image_saver.target = self.ev.target
        self.image_saver.indexes = self.loader.minibatch_indexes
        self.image_saver.labels = self.loader.minibatch_labels
        self.image_saver.minibatch_class = self.loader.minibatch_class
        self.image_saver.minibatch_size = self.loader.minibatch_size
        self.image_saver.this_time = self.decision.snapshot_time
        self.image_saver.gate_skip = self.decision.just_snapshotted
        self.image_saver.gate_skip_not = [1]

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDTanh(device=device)
        self.gd[-1].link_from(self.image_saver)
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

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotters.SimplePlotter(self, name="mse",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else
                                   self.plt[-2])
            self.plt[-1].gate_block = (self.decision.epoch_ended if not i
                                       else [1])
            self.plt[-1].gate_block_not = [1]
        self.plt[0].clear_plot = True
        # Weights plotter
        # """
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_mx = plotters.Weights2D(self, name="First Layer Weights",
                                         limit=16)
        self.plt_mx.input = self.gd[0].weights
        self.plt_mx.input_field = "v"
        self.plt_mx.get_shape_from = self.forward[0].input
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = self.decision.epoch_ended
        self.plt_mx.gate_block_not = [1]
        # """
        # Image plotter
        self.decision.vectors_to_sync[self.forward[0].input] = 1
        self.decision.vectors_to_sync[self.forward[-1].output] = 1
        self.decision.vectors_to_sync[self.ev.target] = 1
        self.plt_img = plotters.Image(self, name="output sample")
        self.plt_img.inputs.append(self.decision)
        self.plt_img.input_fields.append("sample_input")
        self.plt_img.inputs.append(self.decision)
        self.plt_img.input_fields.append("sample_output")
        self.plt_img.inputs.append(self.decision)
        self.plt_img.input_fields.append("sample_target")
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_block = self.decision.epoch_ended
        self.plt_img.gate_block_not = [1]
        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(0, 3):
            self.plt_max.append(plotters.SimplePlotter(self, name="mse",
                                                       plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.plt[-1] if not i else
                                       self.plt_max[-2])
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(0, 3):
            self.plt_min.append(plotters.SimplePlotter(self, name="mse",
                                                       plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.plt_max[-1] if not i else
                                       self.plt_min[-2])
        self.plt_min[-1].redraw_plot = True

    def initialize(self, global_alpha, global_lambda):
        for gd in self.gd:
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
        return super(Workflow, self).initialize()


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    device = opencl.Device()
    w = Workflow(layers=[784, 784], device=device)
    w.initialize(global_alpha=0.001, global_lambda=0.00005)
    w.run()

    plotters.Graphics().wait_finish()
    logging.info("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
