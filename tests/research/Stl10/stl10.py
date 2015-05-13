"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 12, 2015

Workflow for recognition of objects with STL-10 database.

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


import os
import sys

from veles.config import root
from veles.znicz.standard_workflow import StandardWorkflow

sys.path.append(os.path.dirname(__file__))
from .loader_stl import STL10FullBatchLoader  # pylint: disable=W0611


class Stl10(StandardWorkflow):
    """
    Workflow for recognition of objects with STL-10 database.
    """
    def create_workflow(self):
        self.link_repeater(self.start_point)
        self.link_loader(self.repeater)
        self.link_forwards(("input", "minibatch_data"), self.loader)
        self.link_evaluator(self.forwards[-1])
        self.link_decision(self.evaluator)
        end_units = [link(self.decision) for link in
                     (self.link_snapshotter, self.link_error_plotter,
                      self.link_conf_matrix_plotter)]
        end_units.append(self.link_weights_plotter("weights", self.decision))
        self.link_gds(*end_units)
        self.link_loop(self.gds[0])
        self.link_publisher(self.gds[0])
        self.link_end_point(self.publisher)


def run(load, main):
    load(Stl10,
         loader_name=root.stl.loader_name,
         loss_function=root.stl.loss_function,
         loader_config=root.stl.loader,
         layers=root.stl.layers,
         decision_config=root.stl.decision,
         weights_plotter_config=root.stl.weights_plotter,
         publisher_config=root.stl.publisher)
    main()
