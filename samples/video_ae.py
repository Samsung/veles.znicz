#!/usr/bin/python3.3 -O
"""
Created on Mar 20, 2013

File for autoencoding video.

AutoEncoder version.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import logging
import os
import re

from veles.config import root
import veles.plotting_units as plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.znicz.loader as loader


root.defaults = {"decision": {"fail_iterations": 100,
                              "snapshot_prefix": "video_ae"},
                 "loader": {"minibatch_maxsize": 50},
                 "weights_plotter": {"limit": 16},
                 "video_ae": {"global_alpha": 0.0002,
                              "global_lambda": 0.00005,
                              "layers": [9, 14400],
                              "path_for_load_data":
                              os.path.join(root.common.test_dataset_root,
                                           "video/video_ae/img/*.png")}}


class Loader(loader.ImageLoader):
    """Loads dataset.

    Attributes:
        lbl_re_: regular expression for extracting label from filename.
    """
    def init_unpickled(self):
        super(Loader, self).init_unpickled()
        self.lbl_re_ = re.compile("(\d+)\.\w+$")

    def get_label_from_filename(self, filename):
        res = self.lbl_re_.search(filename)
        if res is None:
            return
        lbl = int(res.group(1))
        return lbl


class Workflow(nn_units.NNWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self,
                             train_paths=[root.video_ae.path_for_load_data],
                             minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.repeater)

        # Add forward units
        self.forward = []
        for i in range(0, len(layers)):
            aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                     device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self)
        self.image_saver.link_from(self.forward[-1])

        self.image_saver.link_attrs(self.forward[-1], "output")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indexes", "minibatch_indexes"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")
        self.image_saver.target = self.image_saver.input

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorMSE(self, device=device)
        self.ev.link_from(self.image_saver)
        self.ev.link_attrs(self.forward[-1], ("y", "output"))
        self.ev.link_attrs(self.loader,
                           ("batch_size", "minibatch_size"),
                           ("target", "minibatch_data"),
                           ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.Decision(
            self,
            snapshot_prefix=root.decision.snapshot_prefix,
            fail_iterations=root.decision.fail_iterations)
        self.decision.link_from(self.ev)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "no_more_minibatches_left",
                                 "class_samples")
        self.decision.link_attrs(
            self.ev,
            ("minibatch_metrics", "metrics"))
        self.image_saver.link_attrs(self.decision,
                                    ("this_save_time", "snapshot_time"))
        self.image_saver.gate_skip = ~self.decision.just_snapshotted

        # Add gradient descent units
        self.gd = list(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDTanh(self, device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].link_attrs(self.forward[-1],
                               ("y", "output"),
                               ("h", "input"),
                               "weights", "bias")
        self.gd[-1].link_attrs(self.ev, "err_y")
        self.gd[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gd[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(self, device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].link_attrs(self.forward[i],
                                  ("y", "output"),
                                  ("h", "input"),
                                  "weights", "bias")
            self.gd[i].link_attrs(self.loader, ("batch_size",
                                                "minibatch_size"))
            self.gd[i].link_attrs(self.gd[i + 1], ("err_y", "err_h"))
            self.gd[i].gate_skip = self.decision.gd_skip
        self.repeater.link_from(self.gd[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        """
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_metrics
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        """
        # Matrix plotter
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_mx = plotting_units.Weights2D(
            self, name="First Layer Weights",
            limit=root.weights_plotter.limit)
        self.plt_mx.get_shape_from = self.forward[0].input
        self.plt_mx.input = self.gd[0].weights
        self.plt_mx.input_field = "v"
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended

        """
        # Max plotter
        self.plt_max = []
        styles = ["r--", "b--", "k--"]
        for i in range(0, 3):
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_max[-1].input = self.decision.epoch_metrics
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offs = 1
            self.plt_max[-1].link_from(self.decision)
            self.plt_max[-1].gate_block = ~self.decision.epoch_ended
        # Min plotter
        self.plt_min = []
        styles = ["r:", "b:", "k:"]
        for i in range(0, 3):
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].input = self.decision.epoch_metrics
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offs = 2
            self.plt_min[-1].link_from(self.decision)
            self.plt_min[-1].gate_block = ~self.decision.epoch_ended
        # Image plotter
        self.plt_img = plotting_units.Image(self, name="output sample")
        self.plt_img.inputs.append(self.decision.sample_output)
        self.plt_img.input_fields.append(0)
        self.plt_img.inputs.append(self.decision.sample_input)
        self.plt_img.input_fields.append(0)
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_block = ~self.decision.epoch_ended
        """

    def initialize(self, global_alpha, global_lambda, device):
        self.ev.device = device
        for g in self.gd:
            g.device = device
            g.global_alpha = global_alpha
            g.global_lambda = global_lambda
        for forward in self.forward:
            forward.device = device
        return super(Workflow, self).initialize()


def run(load, main):
    w, snapshot = load(Workflow, layers=root.video_ae.layers)
    if snapshot:
        for forward in w.forward:
            logging.info(forward.weights.v.min(), forward.weights.v.max(),
                         forward.bias.v.min(), forward.bias.v.max())
        w.decision.just_snapshotted << True
    main(global_alpha=root.video_ae.global_alpha,
         global_lambda=root.video_ae.global_lambda)
