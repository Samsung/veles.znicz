#!/usr/bin/python3 -O
"""
Created on Mar 20, 2013

Model created for compress video. Model – autoencoder.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import os
import re

from veles.config import root
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.znicz.loader as loader
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.nn_units import NNSnapshotter
import veles.plotting_units as plotting_units


root.video_ae.update({
    "decision": {"fail_iterations": 100, "max_epochs": 100000},
    "snapshotter": {"prefix": "video_ae"},
    "loader": {"minibatch_size": 50, "on_device": True},
    "weights_plotter": {"limit": 16},
    "learning_rate": 0.000004,
    "weights_decay": 0.00005,
    "layers": [9, [90, 160]],
    "data_paths": os.path.join(root.common.test_dataset_root, "video_ae/img")})


class VideoAELoader(loader.ImageLoader):
    """Loads dataset.

    Attributes:
        lbl_re_: regular expression for extracting label from filename.
    """
    def init_unpickled(self):
        super(VideoAELoader, self).init_unpickled()
        self.lbl_re_ = re.compile("(\d+)\.\w+$")

    def is_valid_filename(self, filename):
        return filename[-4:] == ".png"

    def get_label_from_filename(self, filename):
        res = self.lbl_re_.search(filename)
        if res is None:
            return
        lbl = int(res.group(1))
        return lbl


class VideoAEWorkflow(nn_units.NNWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(VideoAEWorkflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = VideoAELoader(
            self, train_paths=(root.video_ae.data_paths,),
            minibatch_size=root.video_ae.loader.minibatch_size,
            on_device=root.video_ae.loader.on_device)
        self.loader.link_from(self.repeater)

        # Add fwds units
        self.fwds = []
        for i in range(len(layers)):
            aa = all2all.All2AllTanh(self, output_shape=layers[i],
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

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self)
        self.image_saver.link_from(self.fwds[-1])

        self.image_saver.link_attrs(self.fwds[-1], "output")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indexes", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")
        self.image_saver.target = self.image_saver.input

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorMSE(self, device=device)
        self.evaluator.link_from(self.image_saver)
        self.evaluator.link_attrs(self.fwds[-1], "output")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("target", "minibatch_data"))

        # Add decision unit
        self.decision = decision.DecisionMSE(
            self,
            fail_iterations=root.video_ae.decision.fail_iterations,
            max_epochs=root.video_ae.decision.max_epochs)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "minibatch_size",
                                 "last_minibatch",
                                 "class_lengths",
                                 "epoch_ended",
                                 "epoch_number")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_metrics", "metrics"))

        self.snapshotter = NNSnapshotter(
            self, prefix=root.video_ae.snapshotter.prefix,
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
        self.gds = list(None for i in range(0, len(self.fwds)))
        self.gds[-1] = gd.GDTanh(self, device=device)
        self.gds[-1].link_attrs(self.fwds[-1], "output", "input",
                                "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.fwds[i], "output", "input",
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader, ("batch_size",
                                                 "minibatch_size"))
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.gds[0].need_err_input = False
        self.repeater.link_from(self.gds[0])

        # MSE plotter
        prev = self.snapshotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(2, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_metrics"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(prev)
            self.plt[-1].gate_skip = ~self.decision.epoch_ended
            prev = self.plt[-1]
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True

        # Weights plotter
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="First Layer Weights",
            limit=root.video_ae.weights_plotter.limit)
        self.plt_mx.link_attrs(self.fwds[0], ("input", "weights"),
                               ("get_shape_from", "input"))
        self.plt_mx.input_field = "mem"
        self.plt_mx.link_from(prev)
        self.plt_mx.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_mx

        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(0, 3):
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="max min mse", plot_style=styles[i]))
            self.plt_max[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offset = 1
            self.plt_max[-1].link_from(prev)
            self.plt_max[-1].gate_skip = ~self.decision.epoch_ended
            prev = self.plt_max[-1]
        self.plt_max[0].clear_plot = True
        self.plt_max[-1].redraw_plot = True

        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(0, 3):
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="max min mse", plot_style=styles[i]))
            self.plt_min[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offset = 2
            self.plt_min[-1].link_from(prev)
            self.plt_min[-1].gate_skip = ~self.decision.epoch_ended
            prev = self.plt_min[-1]
        self.plt_min[0].clear_plot = True
        self.plt_min[-1].redraw_plot = True

        # Image plotter
        self.plt_img = plotting_units.ImagePlotter(self, name="output sample")
        self.plt_img.inputs.append(self.fwds[-1].output)
        self.plt_img.input_fields.append(0)
        self.plt_img.inputs.append(self.fwds[0].input)
        self.plt_img.input_fields.append(0)
        self.plt_img.link_from(prev)
        self.plt_img.gate_skip = ~self.decision.epoch_ended
        prev = self.plt_img

        self.gds[-1].link_from(prev)
        self.end_point.link_from(prev)
        self.end_point.gate_block = ~self.decision.complete
        self.gds[-1].gate_block = self.decision.complete

    def initialize(self, learning_rate, weights_decay, device, **kwargs):
        self.evaluator.device = device
        for g in self.gds:
            g.device = device
            g.learning_rate = learning_rate
            g.weights_decay = weights_decay
        for forward in self.fwds:
            forward.device = device
        return super(VideoAEWorkflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            device=device)


def run(load, main):
    w, snapshot = load(VideoAEWorkflow, layers=root.video_ae.layers)
    if snapshot:
        for fwds in w.fwds:
            logging.info(fwds.weights.mem.min(), fwds.weights.mem.max(),
                         fwds.bias.mem.min(), fwds.bias.mem.max())
        w.decision.improved <<= True
    main(learning_rate=root.video_ae.learning_rate,
         weights_decay=root.video_ae.weights_decay)
