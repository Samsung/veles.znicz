# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 14, 2014

Model created for object recognition (logotypes of TV channels).
Dataset - Channels. Self-constructing Model. It means that Model can change for
any Model (Convolutional, Fully connected, different parameters)
in configuration file.

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


class ChannelsWorkflow(StandardWorkflow):
    """
    Model created for object recognition (logotypes of TV channels).
Dataset - Channels. Self-constructing Model. It means that Model can change for
any Model (Convolutional, Fully connected, different parameters)
in configuration file.
    """
    def create_workflow(self):
        self.link_downloader(self.start_point)
        self.link_repeater(self.downloader)
        self.link_loader(self.repeater)
        self.link_forwards(("input", "minibatch_data"), self.loader)
        self.link_evaluator(self.forwards[-1])
        self.link_decision(self.evaluator)
        end_units = [link(self.decision) for link in (
            self.link_snapshotter, self.link_error_plotter,
            self.link_conf_matrix_plotter)]
        self.link_image_saver(*end_units)
        last_gd = self.link_gds(self.image_saver)
        self.link_loop(last_gd)
        self.link_end_point(last_gd)


def run(load, main):
    load(ChannelsWorkflow,
         decision_config=root.channels.decision,
         snapshotter_config=root.channels.snapshotter,
         image_saver_config=root.channels.image_saver,
         loader_config=root.channels.loader,
         downloader_config=root.channels.downloader,
         layers=root.channels.layers,
         loader_name=root.channels.loader_name,
         loss_function=root.channels.loss_function)
    main()
