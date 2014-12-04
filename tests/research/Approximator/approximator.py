#!/usr/bin/python3 -O
"""
Created on June 29, 2013

Model created for functions approximation.
Dataset - matlab files with x and y points.
Model - fully-connected Neural Network with MSE loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import scipy.io

from veles.config import root
import veles.error as error
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader
from veles.znicz.nn_units import NNSnapshotter

target_dir = [os.path.join(root.common.test_dataset_root,
                           "approximator/all_org_appertures.mat")]
train_dir = [os.path.join(root.common.test_dataset_root,
                          "approximator/all_dec_appertures.mat")]

root.approximator.update({
    "decision": {"fail_iterations": 1000, "max_epochs": 1000000000},
    "snapshotter": {"prefix": "approximator"},
    "loader": {"minibatch_size": 100},
    "learning_rate": 0.0001,
    "weights_decay": 0.00005,
    "layers": [810, 9],
    "data_paths": {"target": target_dir, "train": train_dir}})


class ApproximatorLoader(loader.ImageLoaderMSE):
    def load_original(self, fnme):
        a = scipy.io.loadmat(fnme)
        for key in a.keys():
            if key[0] != "_":
                vle = a[key]
                break
        else:
            raise error.BadFormatError("Could not find variable to import "
                                       "in %s" % (fnme))
        aa = numpy.zeros(vle.shape, dtype=opencl_types.dtypes[
            root.common.precision_type])
        aa[:] = vle[:]
        return (aa, [])

    def load_data(self):
        super(ApproximatorLoader, self).load_data()
        return
        if self.class_lengths[1] == 0:
            n = self.class_lengths[2] * 10 // 70
            self.class_lengths[1] = n
            self.class_lengths[2] -= n

    def initialize(self, device, **kwargs):
        super(ApproximatorLoader, self).initialize(device, **kwargs)
        self.info("data range: (%.6f, %.6f), target range: (%.6f, %.6f)"
                  % (self.original_data.min(), self.original_data.max(),
                     self.original_targets.min(), self.original_targets.max()))
        # Normalization
        for i in range(0, self.original_data.shape[0]):
            data = self.original_data[i]
            data /= 127.5
            data -= 1.0
            m = data.mean()
            data -= m
            data *= 0.5
            target = self.original_targets[i]
            target /= 127.5
            target -= 1.0
            target -= m
            target *= 0.5

        self.info("norm data range: (%.6f, %.6f), "
                  "norm target range: (%.6f, %.6f)"
                  % (self.original_data.min(), self.original_data.max(),
                     self.original_targets.min(), self.original_targets.max()))


class ApproximatorWorkflow(nn_units.NNWorkflow):
    """
    Model created for functions approximation. Dataset - matlab files with x
    and y points. Model - fully-connected Neural Network with MSE loss
    function.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(ApproximatorWorkflow, self).__init__(workflow, **kwargs)
        self.repeater.link_from(self.start_point)

        self.loader = ApproximatorLoader(
            self, train_paths=root.approximator.data_paths.train,
            target_paths=root.approximator.data_paths.target)
        self.loader.link_from(self.repeater)

        # Add fwds units
        self.forwards = []
        for i in range(0, len(layers)):
            aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                     device=device)
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
        self.evaluator = evaluator.EvaluatorMSE(self, device=device)
        self.evaluator.link_from(self.forwards[-1])
        self.evaluator.link_attrs(self.forwards[-1], "output")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("target", "minibatch_targets"))

        # Add decision unit
        self.decision = decision.DecisionMSE(
            self, fail_iterations=root.approximator.decision.fail_iterations,
            max_epochs=root.approximator.decision.max_epochs)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "last_minibatch",
                                 "class_lengths",
                                 "epoch_ended",
                                 "epoch_number",
                                 "minibatch_offset",
                                 "minibatch_size")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_mse", "mse"),
            ("minibatch_metrics", "metrics"))

        self.snapshotter = NNSnapshotter(
            self, prefix=root.approximator.snapshotter.prefix,
            directory=root.common.snapshot_dir)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.decision.epoch_ended | ~self.decision.improved)

        # Add gradient descent units
        self.gds = list(None for i in range(0, len(self.forwards)))
        self.gds[-1] = gd.GDTanh(self, device=device)
        self.gds[-1].link_from(self.snapshotter)
        self.gds[-1].link_attrs(self.forwards[-1], "output", "input",
                                "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.forwards) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device)
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

        # Average plotter
        self.plt_avg = []
        styles = ["", "b-", "k-"]  # ["r-", "b-", "k-"]
        j = 0
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_avg.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_avg[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_avg[-1].input_field = i
            self.plt_avg[-1].link_from(self.plt_avg[-2] if j
                                       else self.decision)
            self.plt_avg[-1].gate_block = (Bool(False) if j
                                           else ~self.decision.epoch_ended)
            j += 1
        self.plt_avg[0].clear_plot = True

        # Max plotter
        self.plt_max = []
        styles = ["", "b--", "k--"]  # ["r--", "b--", "k--"]
        j = 0
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_max[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offset = 1
            self.plt_max[-1].link_from(self.plt_max[-2] if j
                                       else self.plt_avg[-1])
            j += 1

        # Min plotter
        self.plt_min = []
        styles = ["", "b:", "k:"]  # ["r:", "b:", "k:"]
        j = 0
        for i in range(0, len(styles)):
            if not len(styles[i]):
                continue
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offset = 2
            self.plt_min[-1].link_from(self.plt_min[-2] if j
                                       else self.plt_max[-1])
            j += 1
        self.plt_min[-1].redraw_plot = True

        # ImmediatePlotter
        self.plt = plotting_units.ImmediatePlotter(
            self, name="ImmediatePlotter", ylim=[-1.1, 1.1])
        del self.plt.inputs[:]
        self.plt.inputs.append(self.loader.minibatch_data)
        self.plt.inputs.append(self.loader.minibatch_targets)
        self.plt.inputs.append(self.forwards[-1].output)
        del self.plt.input_fields[:]
        self.plt.input_fields.append(0)
        self.plt.input_fields.append(0)
        self.plt.input_fields.append(0)
        del self.plt.input_styles[:]
        self.plt.input_styles.append("k-")
        self.plt.input_styles.append("g-")
        self.plt.input_styles.append("b-")
        self.plt.link_from(self.decision)
        self.plt.gate_block = ~self.decision.epoch_ended

    def initialize(self, learning_rate, weights_decay, minibatch_size,
                   device, **kwargs):
        super(ApproximatorWorkflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            minibatch_size=minibatch_size, device=device)


def run(load, main):
    load(ApproximatorWorkflow, layers=root.approximator.layers)
    main(learning_rate=root.approximator.learning_rate,
         weights_decay=root.approximator.weights_decay,
         minibatch_size=root.approximator.loader.minibatch_size)
