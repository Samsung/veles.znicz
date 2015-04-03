# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Nov 13, 2014

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


class YaleFacesWorkflow(StandardWorkflow):
    """
    Model was created for face recognition. Database - Yale Faces.
    Model - fully-connected Neural Network with SoftMax loss function.
    """

    def link_end_point(self, init_unit):
        self.end_point.link_from(init_unit)
        self.end_point.gate_block = ~self.loader.train_ended
        self.loader.gate_block = self.loader.train_ended

    def create_workflow(self):
        self.link_repeater(self.start_point)
        self.link_loader(self.repeater)
        self.link_data_saver(self.loader)
        self.link_repeater(self.data_saver)
        self.link_end_point(self.repeater)


def run(load, main):
    load(YaleFacesWorkflow,
         loader_name=root.yalefaces.loader_name,
         data_saver_config=root.yalefaces.datasaver,
         preprocessing=root.yalefaces.preprocessing,
         loader_config=root.yalefaces.loader)
    main()
