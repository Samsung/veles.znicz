# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mar 20, 2013

Model created for compress video. Model - autoencoder.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import logging

from zope.interface import implementer

from veles.config import root
from veles.normalization import NoneNormalizer
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
from veles.znicz.downloader import Downloader
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.loader as loader
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.nn_units import NNSnapshotter
import veles.plotting_units as plotting_units


@implementer(loader.IFileLoader)
class VideoAELoader(loader.FullBatchAutoLabelFileImageLoader):
    """Loads dataset.

    Attributes:
        lbl_re_: regular expression for extracting label from filename.
    """
    def __init__(self, workflow, **kwargs):
        super(VideoAELoader, self).__init__(
            workflow, label_regexp="(\\d+)\\.\\w+$",
            file_subtypes="png", **kwargs)
        self.target_normalizer = NoneNormalizer


class VideoAEWorkflow(nn_units.NNWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        kwargs["layers"] = layers
        super(VideoAEWorkflow, self).__init__(workflow, **kwargs)

        self.downloader = Downloader(
            self, url=root.video_ae.downloader.url,
            directory=root.video_ae.downloader.directory,
            files=root.video_ae.downloader.files)
        self.downloader.link_from(self.start_point)

        self.repeater.link_from(self.downloader)

        self.loader = VideoAELoader(
            self, **root.video_ae.loader.__content__)
        self.loader.link_from(self.repeater)

        # Add fwds units
        for i in range(len(layers)):
            aa = all2all.All2AllTanh(self, output_sample_shape=layers[i])
            self.forwards.append(aa)
            if i:
                self.forwards[i].link_from(self.forwards[i - 1])
                self.forwards[i].link_attrs(
                    self.forwards[i - 1], ("input", "output"))
            else:
                self.forwards[i].link_from(self.loader)
                self.forwards[i].link_attrs(
                    self.loader, ("input", "minibatch_data"))

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self)
        self.image_saver.link_from(self.forwards[-1])

        self.image_saver.link_attrs(self.forwards[-1], "output")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indices", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")
        self.image_saver.target = self.image_saver.input

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorMSE(self)
        self.evaluator.link_from(self.image_saver)
        self.evaluator.link_attrs(self.forwards[-1], "output")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("normalizer", "target_normalizer"),
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
        self.snapshotter.gate_skip = ~self.loader.epoch_ended
        self.snapshotter.skip = ~self.decision.improved

        self.image_saver.gate_skip = ~self.decision.improved
        self.image_saver.link_attrs(self.snapshotter,
                                    ("this_save_time", "time"))

        # Add gradient descent units
        self.gds[:] = [None] * len(self.forwards)

        self.gds[-1] = gd.GDTanh(self)
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
        self.plt_mx.link_attrs(self.forwards[0], ("input", "weights"),
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
        self.plt_img.inputs.append(self.forwards[-1].output)
        self.plt_img.input_fields.append(0)
        self.plt_img.inputs.append(self.forwards[0].input)
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
        for forward in self.forwards:
            forward.device = device
        return super(VideoAEWorkflow, self).initialize(
            learning_rate=learning_rate, weights_decay=weights_decay,
            device=device, **kwargs)


def run(load, main):
    w, snapshot = load(VideoAEWorkflow, layers=root.video_ae.layers)
    if snapshot:
        for fwds in w.fwds:
            logging.info(fwds.weights.mem.min(), fwds.weights.mem.max(),
                         fwds.bias.mem.min(), fwds.bias.mem.max())
        w.decision.improved <<= True
    main(learning_rate=root.video_ae.learning_rate,
         weights_decay=root.video_ae.weights_decay)
