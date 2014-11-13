#!/usr/bin/python3 -O
"""
Created on Jul 3, 2013

Model created for object recognition. Dataset – CIFAR10. Self-constructing
Model. It means that Model can change for any Model (Convolutional, Fully
connected, different parameters) in configuration file.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import pickle
import six
from zope.interface import implementer

from veles.config import root
import veles.formats as formats
import veles.plotting_units as plotting_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.conv as conv
import veles.znicz.evaluator as evaluator
import veles.znicz.image_saver as image_saver
import veles.znicz.lr_adjust as lr_adjust
import veles.znicz.loader as loader
import veles.znicz.diversity as diversity
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.nn_units import NNSnapshotter
from veles.znicz.standard_workflow import StandardWorkflow


train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.cifar.update({
    "accumulator": {"bars": 30},
    "decision": {"fail_iterations": 100, "max_epochs": 1000000000},
    "learning_rate_adjust": {"do": False},
    "snapshotter": {"prefix": "cifar"},
    "softmax": {"error_function_avr": False},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 180, "norm": "-1, 1",
               "shuffle_limit": 1, "sobel": False, "on_device": True},
    "weights_plotter": {"limit": 64},
    "similar_weights_plotter": {'form': 1.1, 'peak': 0.5, 'magnitude': 0.65,
                                'layers': {1}},
    "layers": [{"type": "all2all_tanh", "learning_rate": 0.0005,
                "weights_decay": 0.00005, "output_shape": 100},
               {"type": "softmax", "output_shape": 10, "learning_rate": 0.0005,
                "weights_decay": 0.00005}],
    "data_paths": {"train": train_dir, "validation": validation_dir}})


@implementer(loader.IFullBatchLoader)
class CifarLoader(loader.FullBatchLoader):
    """Loads Cifar dataset.
    """
    def __init__(self, workflow, **kwargs):
        super(CifarLoader, self).__init__(workflow, **kwargs)
        self.shuffle_limit = kwargs.get("shuffle_limit", 2000000000)

    def shuffle(self):
        if self.shuffle_limit <= 0:
            return
        self.shuffle_limit -= 1
        self.info("Shuffling, remaining limit is %d", self.shuffle_limit)
        super(CifarLoader, self).shuffle()

    def _add_sobel_chan(self):
        """
        Adds 4th channel (Sobel filtered image) to `self.original_data`
        """
        import cv2

        sobel_data = numpy.zeros(shape=self.original_data.shape[:-1],
                                 dtype=numpy.float32)

        for i in range(self.original_data.shape[0]):
            pic = self.original_data[i, :, :, 0:3]
            sobel_x = cv2.Sobel(pic, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(pic, cv2.CV_32F, 0, 1, ksize=3)
            sobel_total = numpy.sqrt(numpy.square(sobel_x) +
                                     numpy.square(sobel_y))
            sobel_gray = cv2.cvtColor(sobel_total, cv2.COLOR_BGR2GRAY)
            formats.normalize(sobel_gray)

            if root.cifar.loader.norm == "mean":
                sobel_data[i, :, :] = (sobel_gray + 1) / 2 * 255
            elif root.cifar.loader.norm == "-128, 128":
                sobel_data[i, :, :] = sobel_gray * 128
            elif root.cifar.loader.norm == "-1, 1":
                sobel_data[i, :, :] = sobel_gray

        sobel_data = sobel_data.reshape(self.original_data.shape[:-1] + (1,))
        numpy.append(self.original_data, sobel_data, axis=3)

    def load_data(self):
        """Here we will load data.
        """
        self.original_data.mem = numpy.zeros([60000, 32, 32, 3],
                                             dtype=numpy.float32)
        self.original_labels.mem = numpy.zeros(60000, dtype=numpy.int32)

        # Load Validation
        with open(root.cifar.data_paths.validation, "rb") as fin:
            if six.PY3:
                vle = pickle.load(fin, encoding='latin1')
            else:
                vle = pickle.load(fin)
        fin.close()
        self.original_data.mem[:10000] = formats.interleave(
            vle["data"].reshape(10000, 3, 32, 32))[:]
        self.original_labels.mem[:10000] = vle["labels"][:]

        # Load Train
        for i in range(1, 6):
            with open(os.path.join(root.cifar.data_paths.train,
                                   ("data_batch_%d" % i)), "rb") as fin:
                if six.PY3:
                    vle = pickle.load(fin, encoding='latin1')
                else:
                    vle = pickle.load(fin)
            self.original_data.mem[i * 10000: (i + 1) * 10000] = (
                formats.interleave(vle["data"].reshape(10000, 3, 32, 32))[:])
            self.original_labels.mem[i * 10000: (i + 1) * 10000] = (
                vle["labels"][:])

        self.class_lengths[0] = 0
        self.class_offsets[0] = 0
        self.class_lengths[1] = 10000
        self.class_offsets[1] = 10000
        self.class_lengths[2] = 50000
        self.class_offsets[2] = 60000

        self.total_samples = self.original_data.shape[0]

        use_sobel = root.cifar.loader.sobel
        if use_sobel:
            self._add_sobel_chan()

        if root.cifar.loader.norm == "mean":
            mean = numpy.mean(self.original_data[10000:], axis=0)
            self.original_data.mem -= mean
            self.info("Validation range: %.6f %.6f %.6f",
                      self.original_data.mem[:10000].min(),
                      self.original_data.mem[:10000].max(),
                      numpy.average(self.original_data.mem[:10000]))
            self.info("Train range: %.6f %.6f %.6f",
                      self.original_data.mem[10000:].min(),
                      self.original_data.mem[10000:].max(),
                      numpy.average(self.original_data.mem[10000:]))
        elif root.cifar.loader.norm == "-1, 1":
            for sample in self.original_data.mem:
                formats.normalize(sample)
        elif root.cifar.loader.norm == "-128, 128":
            for sample in self.original_data.mem:
                formats.normalize(sample)
                sample *= 128
        else:
            raise ValueError("Unsupported normalization type "
                             + str(root.cifar.loader.norm))


class CifarWorkflow(StandardWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        kwargs["layers"] = layers
        super(CifarWorkflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = CifarLoader(
            self, shuffle_limit=root.cifar.loader.shuffle_limit,
            on_device=root.cifar.loader.on_device)
        self.loader.link_from(self.repeater)

        # Add fwds units
        self.parse_forwards_from_config()

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(
            self, out_dirs=root.cifar.image_saver.out_dirs)
        self.image_saver.link_from(self.fwds[-1])
        self.image_saver.link_attrs(self.fwds[-1],
                                    "output", "max_idx")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indexes", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self)
        self.evaluator.link_from(self.image_saver)
        self.evaluator.link_attrs(self.fwds[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("labels", "minibatch_labels"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.DecisionGD(
            self, fail_iterations=root.cifar.decision.fail_iterations,
            max_epochs=root.cifar.decision.max_epochs)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "last_minibatch",
                                 "minibatch_size",
                                 "class_lengths",
                                 "epoch_ended",
                                 "epoch_number")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_output_sum"))

        self.snapshotter = NNSnapshotter(
            self, prefix=root.cifar.snapshotter.prefix,
            directory=root.common.snapshot_dir, compress="")
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.decision.epoch_ended | ~self.decision.improved)

        self.image_saver.gate_skip = ~self.decision.improved
        self.image_saver.link_attrs(self.snapshotter,
                                    ("this_save_time", "time"))

        # Add gradient descent units
        self.create_gd_units_by_config()

        if root.learning_rate_adjust.do:
            # Add learning_rate_adjust unit
            lr_adjuster = lr_adjust.LearningRateAdjust(self)
            for gd_elm in self.gds:
                lr_adjuster.add_gd_unit(
                    gd_elm,
                    lr_policy=lr_adjust.ArbitraryStepPolicy(
                        [(gd_elm.learning_rate, 60000),
                         (gd_elm.learning_rate / 10., 5000),
                         (gd_elm.learning_rate / 100., 100000000)]),
                    bias_lr_policy=lr_adjust.ArbitraryStepPolicy(
                        [(gd_elm.learning_rate_bias, 60000),
                         (gd_elm.learning_rate_bias / 10., 5000),
                         (gd_elm.learning_rate_bias / 100., 100000000)])
                    )
            lr_adjuster.link_from(self.gds[0])
            self.repeater.link_from(lr_adjuster)
            self.end_point.link_from(lr_adjuster)
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
        self.plt_mx = []
        for i in range(1, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].link_attrs(self.decision,
                                       ("input", "confusion_matrixes"))
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended

        # Err y plotter
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
        self.plt_wd = []
        prev_channels = 3
        for i in range(len(layers)):
            if (not isinstance(self.fwds[i], conv.Conv) and
                    not isinstance(self.fwds[i], all2all.All2All)):
                continue
            plt_mx = nn_plotting_units.Weights2D(
                self, name="%s %s" % (i + 1, layers[i]["type"]),
                limit=root.cifar.weights_plotter.limit)
            self.plt_wd.append(plt_mx)
            self.plt_wd[-1].link_attrs(self.fwds[i], ("input", "weights"))
            self.plt_wd[-1].input_field = "mem"
            if isinstance(self.fwds[i], conv.Conv):
                self.plt_wd[-1].get_shape_from = (
                    [self.fwds[i].kx, self.fwds[i].ky, prev_channels])
                prev_channels = self.fwds[i].n_kernels
            if (layers[i].get("output_shape") is not None and
                    layers[i]["type"] != "softmax"):
                self.plt_wd[-1].link_attrs(self.fwds[i],
                                           ("get_shape_from", "input"))
            self.plt_wd[-1].link_from(self.decision)
            self.plt_wd[-1].gate_block = ~self.decision.epoch_ended
            magnitude = root.cifar.similar_weights_plotter.magnitude
            if isinstance(self.fwds[i], conv.Conv) and \
               (i + 1) in root.cifar.similar_weights_plotter.layers:
                plt_mx = diversity.SimilarWeights2D(
                    self, name="%s %s [similar]" % (i + 1, layers[i]["type"]),
                    limit=root.cifar.weights_plotter.limit,
                    form_threshold=root.cifar.similar_weights_plotter.form,
                    peak_threshold=root.cifar.similar_weights_plotter.peak,
                    magnitude_threshold=magnitude)
                self.plt_wd.append(plt_mx)
                self.plt_wd[-1].link_attrs(self.fwds[i], ("input", "weights"))
                self.plt_wd[-1].input_field = "mem"
                self.plt_wd[-1].get_shape_from = self.plt_wd[-2].get_shape_from
                if (layers[i].get("output_shape") is not None and
                        layers[i]["type"] != "softmax"):
                    self.plt_wd[-1].link_attrs(self.fwds[i],
                                               ("get_shape_from", "input"))
                self.plt_wd[-1].link_from(self.decision)
                self.plt_wd[-1].gate_block = ~self.decision.epoch_ended
        self.plt_wd[0].clear_plot = True
        self.plt_wd[-1].redraw_plot = True

        # Table plotter
        self.plt_tab = plotting_units.TableMaxMin(self, name="Max, Min")
        del self.plt_tab.y[:]
        del self.plt_tab.col_labels[:]
        for i in range(0, len(layers)):
            if (not isinstance(self.fwds[i], conv.Conv) and
                    not isinstance(self.fwds[i], all2all.All2All)):
                continue
            obj = self.fwds[i].weights
            name = "weights %s %s" % (i + 1, layers[i]["type"])
            self.plt_tab.y.append(obj)
            self.plt_tab.col_labels.append(name)
            obj = self.gds[i].gradient_weights
            name = "gd %s %s" % (i + 1, layers[i]["type"])
            self.plt_tab.y.append(obj)
            self.plt_tab.col_labels.append(name)
            obj = self.fwds[i].output
            name = "Y %s %s" % (i + 1, layers[i]["type"])
            self.plt_tab.y.append(obj)
            self.plt_tab.col_labels.append(name)
        self.plt_tab.link_from(self.decision)
        self.plt_tab.gate_block = ~self.decision.epoch_ended

        self.gds[-1].unlink_before()
        self.gds[-1].link_from(self.snapshotter)

    def initialize(self, device, **kwargs):
        super(CifarWorkflow, self).initialize(device, **kwargs)


def run(load, main):
    load(CifarWorkflow, layers=root.cifar.layers)
    main(minibatch_size=root.cifar.loader.minibatch_size)
