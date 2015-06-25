# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 6, 2014

Model created for geometric figure recognition. Dataset was synthetically
generated by VELES. Self-constructing Model. It means that Model can change
for any Model (Convolutional, Fully connected, different parameters) in
configuration file.

A workflow to test first layer in simple line detection.

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


from veles.config import root
from veles.znicz.standard_workflow import StandardWorkflow


class LinesWorkflow(StandardWorkflow):
    """
    Model created for geometric figure recognition. Dataset was synthetically
    generated by VELES. You can use draw_lines.py to generate the dataset
    or download it from the specified URL.
    """
    def create_workflow(self):
        self.link_downloader(self.start_point)
        self.link_repeater(self.downloader)
        self.link_loader(self.repeater)
        self.link_forwards(("input", "minibatch_data"), self.loader)
        self.link_evaluator(self.forwards[-1])
        self.link_decision(self.evaluator)
        end_units = [link(self.decision) for link in (self.link_snapshotter,
                                                      self.link_error_plotter)]
        self.link_image_saver(*end_units)
        gd = self.link_gds(self.image_saver)
        self.link_table_plotter(gd).gate_block = self.decision.complete
        last_weights = self.link_weights_plotter(
            "gradient_weights", self.table_plotter)
        self.link_multi_hist_plotter(
            "gradient_weights", last_weights)
        self.repeater.link_from(self.multi_hist_plotter[-1])

        self.link_end_point(gd)


def run(load, main):
    load(LinesWorkflow,
         decision_config=root.lines.decision,
         snapshotter_config=root.lines.snapshotter,
         image_saver_config=root.lines.image_saver,
         loader_config=root.lines.loader,
         loader_name=root.lines.loader_name,
         loss_function=root.lines.loss_function,
         downloader_config=root.lines.downloader,
         weights_plotter_config=root.lines.weights_plotter,
         mcdnnic_topology=root.lines.mcdnnic_topology,
         mcdnnic_parameters=root.lines.mcdnnic_parameters)
    main()
