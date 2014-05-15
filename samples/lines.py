#!/usr/bin/python3.3 -O
# encoding: utf-8
"""
Created on May 6, 2014

A workflow to test first layer in simple line detection.
"""


import os

from veles.config import root
from veles.mutable import Bool
from veles.znicz import conv, all2all, evaluator, decision
from veles.znicz.standard_workflow import StandardWorkflow
from veles.znicz.loader import ImageLoader
import veles.znicz.image_saver as image_saver
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.plotting_units as plotting_units
from enum import IntEnum

train = "/data/veles/Lines/Grid/learn"
valid = "/data/veles/Lines/Grid/test"

root.model = "grid"

root.defaults = {"decision": {"fail_iterations": 100,
                              "snapshot_prefix": "lines"},
                 "loader": {"minibatch_maxsize": 60},
                 "weights_plotter": {"limit": 32},
                 "image_saver": {"out_dirs":
                                 [os.path.join(root.common.cache_dir,
                                               "tmp %s/test" % root.model),
                                  os.path.join(root.common.cache_dir,
                                               "tmp %s/validation" %
                                               root.model),
                                  os.path.join(root.common.cache_dir,
                                               "tmp %s/train" % root.model)]},
                 "lines": {"learning_rate": 0.01, "weights_decay": 0.0,
                           "layers":
                           [{"type": "conv_relu", "n_kernels": 32,
                             "kx": 13, "ky": 13,
                             "sliding": (2, 2), "padding": (1, 1, 1, 1),
                             "learning_rate": 0.01, "learning_rate_bias": 0.02,
                             "gradient_moment": 0.9,
                             "weights_filling": "gaussian",
                             "weights_stddev": 0.0001,
                             "bias_filling": "constant", "bias_stddev": 0.0001,
                             "weights_decay": 0.0, "weights_decay_bias": 0.0},
                            {"type": "max_pooling",
                             "kx": 5, "ky": 5, "sliding": (2, 2)},
                            {"type": "avg_pooling",
                             "kx": 5, "ky": 5, "sliding": (2, 2)},
                            {"type": "norm",
                             "alpha": 0.00005, "beta": 0.75, "n": 3},
                            {"type": "conv_relu", "n_kernels": 32,
                             "kx": 7, "ky": 7,
                             "sliding": (1, 1), "padding": (2, 2, 2, 2),
                             "learning_rate": 0.01, "learning_rate_bias": 0.02,
                             "gradient_moment": 0.9,
                             "weights_filling": "gaussian",
                             "weights_stddev": 0.01,
                             "bias_filling": "constant", "bias_stddev": 0.01,
                             "weights_decay": 0.0, "weights_decay_bias": 0.0},
                            {"type": "avg_pooling",
                             "kx": 3, "ky": 3, "sliding": (2, 2)},
                            {"type": "norm",
                             "alpha": 0.00005, "beta": 0.75, "n": 3},
                            {"type": "softmax", "output_shape": 6,
                             "gradient_moment": 0.9,
                             "weights_filling": "uniform",
                             "weights_stddev": 0.05,
                             "bias_filling": "constant", "bias_stddev": 0.05,
                             "learning_rate": 0.01, "learning_rate_bias": 0.02,
                             "weights_decay": 1, "weights_decay_bias": 0}],
                           "path_for_load_data": {"validation": valid,
                                                  "train": train}}}


class ImageLabel(IntEnum):
    """An enum for different figure primitive classes"""
    vertical = 0  # |
    horizontal = 1  # --
    tilted_bottom_to_top = 2  # left lower --> right top (/)
    tilted_top_to_bottom = 3  # left top --> right bottom (\)
    straight_grid = 4  # 0 and 90 deg lines simultaneously
    tilted_grid = 5  # +45 and -45 deg lines simultaneously
    #circle = 6
    #square = 7
    #right_angle = 8
    #triangle = 9
    #sinusoid = 10


class Loader(ImageLoader):
    def get_label_from_filename(self, filename):
        #takes folder name "vertical", "horizontal", "etc"
        return int(ImageLabel[filename.split("/")[-2]])


class Workflow(StandardWorkflow):
    """Workflow for Lines dataset.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        kwargs["name"] = kwargs.get("name", "Lines")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)
        self.loader = Loader(
            self,
            train_paths=[root.lines.path_for_load_data.train],
            validation_paths=[root.lines.path_for_load_data.validation],
            minibatch_maxsize=root.loader.minibatch_maxsize,
            grayscale=False)

        self.loader.load_data()
        self.loader.link_from(self.repeater)

        self._parse_forwards_from_config()

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(
            self, out_dirs=root.image_saver.out_dirs)
        self.image_saver.link_from(self.fwds[-1])
        self.image_saver.link_attrs(self.fwds[-1], "output", "max_idx")
        self.image_saver.link_attrs(
            self.loader,
            ("input", "minibatch_data"),
            ("indexes", "minibatch_indexes"),
            ("labels", "minibatch_labels"),
            "minibatch_class", "minibatch_size")

        # EVALUATOR
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.image_saver)
        self.evaluator.link_attrs(self.fwds[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("labels", "minibatch_labels"))

        # Add decision unit
        self.decision = decision.Decision(
            self, fail_iterations=root.decision.fail_iterations,
            snapshot_prefix=root.decision.snapshot_prefix)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "class_samples",
                                 "no_more_minibatches_left")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_output_sum"))

        # BACKWARD LAYERS (GRADIENT DESCENT)
        self._create_gradient_descent_units()
        """
        # Weights plotter
        self.decision.vectors_to_sync[self.gds[0].weights] = 1
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit)
        self.plt_mx.link_attrs(self.gds[0], ("input", "weights"))
        self.plt_mx.input_field = "v"
        self.plt_mx.get_shape_from = (
            [self.fwds[0].kx, self.fwds[0].ky, 3]
            if isinstance(self.fwds[0], conv.Conv)
            else self.fwds[0].input)
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended
        """

        # Weights plotter
        self.plt_mx = []
        prev_channels = 3
        for i in range(0, len(layers)):
            if (not isinstance(self.fwds[i], conv.Conv) and
                    not isinstance(self.fwds[i], all2all.All2All)):
                continue
            self.decision.vectors_to_sync[self.fwds[i].weights] = 1
            plt_mx = nn_plotting_units.Weights2D(
                self, name="%s %s" % (i + 1, layers[i]["type"]),
                limit=root.weights_plotter.limit)
            self.plt_mx.append(plt_mx)
            self.plt_mx[-1].link_attrs(self.fwds[i], ("input", "weights"))
            self.plt_mx[-1].input_field = "v"
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

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision
                                   if len(self.plt) == 1 else self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended
                                       if len(self.plt) == 1 else Bool(False))
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True

        # MultiHistogram plotter
        self.plt_multi_hist = []
        for i in range(0, len(layers)):
            multi_hist = plotting_units.MultiHistogram(
                self, name="Histogram %s %s" % (i + 1, layers[i]["type"]))
            self.plt_multi_hist.append(multi_hist)
            if layers[i].get("n_kernels") is not None:
                self.plt_multi_hist[i].link_from(self.decision)
                self.plt_multi_hist[i].hist_number = layers[i]["n_kernels"]
                self.plt_multi_hist[i].link_attrs(self.fwds[i],
                                                  ("input", "weights"))
                end_epoch = ~self.decision.epoch_ended
                self.plt_multi_hist[i].gate_block = end_epoch
            if layers[i].get("output_shape") is not None:
                self.plt_multi_hist[i].link_from(self.decision)
                self.plt_multi_hist[i].hist_number = layers[i]["output_shape"]
                self.plt_multi_hist[i].link_attrs(self.fwds[i],
                                                  ("input", "weights"))
                self.plt_multi_hist[i].gate_block = ~self.decision.epoch_ended
        # repeater and gate block
        self.repeater.link_from(self.gds[0])
        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete
        self.loader.gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device):
        #self.gds[0].learning_rate = 0.03
        super(Workflow, self).initialize(learning_rate=learning_rate,
                                         weights_decay=weights_decay,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.lines.layers)
    main(learning_rate=root.lines.learning_rate,
         weights_decay=root.lines.weights_decay)
