#!/usr/bin/python3.3 -O
# encoding: utf-8
"""
Created on May 6, 2014

A workflow to test first layer in simple line detection.
"""


from veles.config import root
import veles.error as error
from veles.mutable import Bool
from veles.znicz import conv, pooling, all2all, evaluator, decision
from veles.znicz.standard_workflow import StandardWorkflow
from veles.znicz.loader import ImageLoader
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.plotting_units as plotting_units
from enum import IntEnum


root.defaults = {"all2all_relu": {"weights_filling": "uniform",
                                  "weights_stddev": 0.05},
                 "conv_relu":  {"weights_filling": "gaussian",
                                "weights_stddev": 0.001},
                 "decision": {"fail_iterations": 100,
                              "snapshot_prefix": "lines"},
                 "loader": {"minibatch_maxsize": 60},
                 "weights_plotter": {"limit": 32},
                 "lines": {"learning_rate": 0.01,
                           "weights_decay": 0.0,
                           "layers":
                           [{"type": "conv_relu", "n_kernels": 32,
                             "kx": 11, "ky": 11, "sliding": (4, 4),
                             "padding": (0, 0, 0, 0)},
                            {"type": "max_pooling",
                             "kx": 3, "ky": 3, "sliding": (2, 2)},
                            {"type": "relu", "layers": 32},
                            {"type": "softmax", "layers": 4}]},
                 "softmax": {"weights_filling": "uniform",
                             "weights_stddev": 0.05}}


class ImageLabel(IntEnum):
    """Enum for different types of lines"""
    vertical = 0
    horizontal = 1
    tilted_bottom_to_top = 2  # left lower --> right top
    tilted_top_to_bottom = 3  # left top --> right bottom


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
            train_paths=[
                "/data/veles/Lines/LINES_10_500_NOISY_min_valid/learning"],
            validation_paths=[
                "/data/veles/Lines/LINES_10_500_NOISY_min_valid/test"],
            minibatch_maxsize=root.loader.minibatch_maxsize,
            grayscale=False)

        self.loader.load_data()

        self.loader.link_from(self.repeater)

        del self.fwds[:]
        for i in range(0, len(layers)):
            layer = layers[i]
            if layer["type"] == "conv_relu":
                aa = conv.ConvRELU(
                    self, n_kernels=layer["n_kernels"],
                    kx=layer["kx"], ky=layer["ky"],
                    sliding=layer.get("sliding", (1, 1, 1, 1)),
                    padding=layer.get("padding", (0, 0, 0, 0)),
                    device=device,
                    weights_filling=root.conv_relu.weights_filling,
                    weights_stddev=root.conv_relu.weights_stddev)
            elif layer["type"] == "max_pooling":
                aa = pooling.MaxPooling(
                    self, kx=layer["kx"], ky=layer["ky"],
                    sliding=layer.get("sliding", (layer["kx"], layer["ky"])),
                    device=device)
            elif layer["type"] == "avg_pooling":
                aa = pooling.AvgPooling(self, kx=layer["kx"], ky=layer["ky"],
                                        sliding=layer.get("sliding",
                                                          (layer["kx"],
                                                           layer["ky"])),
                                        device=device)
            elif layer["type"] == "relu":
                aa = all2all.All2AllRELU(
                    self,
                    weights_filling=root.all2all_relu.weights_filling,
                    weights_stddev=root.all2all_relu.weights_stddev,
                    output_shape=[layer["layers"]], device=device)
            elif layer["type"] == "softmax":
                aa = all2all.All2AllSoftmax(
                    self,
                    output_shape=[layer["layers"]],
                    weights_filling=root.softmax.weights_filling,
                    weights_stddev=root.softmax.weights_stddev,
                    device=device)
            else:
                raise error.ErrBadFormat("Unsupported layer type %s" %
                                         (layer["type"]))
            self.fwds.append(aa)
            if i:
                self.fwds[-1].link_from(self.fwds[-2])
                self.fwds[-1].link_attrs(self.fwds[-2],
                                         ("input", "output"))
            else:
                self.fwds[-1].link_from(self.loader)
                self.fwds[-1].link_attrs(self.loader,
                                         ("input", "minibatch_data"))
            #self._add_forward_unit(aa)

        # EVALUATOR
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.fwds[-1], ("y", "output"), "max_idx")
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
            ("minibatch_max_err_y_sum", "max_err_y_sum"))

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
            if (layers[i].get("layers") is not None and
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
            if layers[i].get("layers") is not None:
                self.plt_multi_hist[i].link_from(self.decision)
                self.plt_multi_hist[i].hist_number = layers[i]["layers"]
                self.plt_multi_hist[i].link_attrs(self.fwds[i],
                                                  ("input", "weights"))
                self.plt_multi_hist[i].gate_block = ~self.decision.epoch_ended
        # repeater and gate block
        self.repeater.link_from(self.gds[0])
        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete
        self.loader.gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device):
        self.gds[0].learning_rate = 0.03
        super(Workflow, self).initialize(learning_rate=learning_rate,
                                         weights_decay=weights_decay,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.lines.layers)
    main(learning_rate=root.lines.learning_rate,
         weights_decay=root.lines.weights_decay)
