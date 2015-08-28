# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May August, 2015

Workflow, which constructs forward and backward runs like in Caffe from
snapshot of trained Model.

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


from veles.mutable import Bool
from veles.snapshotter import SnapshotterToFile
from veles.units import nothing
from veles.znicz.nn_units import NNWorkflow


class LabWorkflow(NNWorkflow):
    """
    Workflow, which constructs forward and backward runs like in Caffe from
    snapshot of trained Model.
    """
    def __init__(self, workflow, **kwargs):
        super(LabWorkflow, self).__init__(workflow, **kwargs)
        self.snapshot_name = kwargs["snapshot_name"]
        self.train_wf = SnapshotterToFile.import_(self.snapshot_name)
        self.train_wf.stopped = True
        self.train_wf.workflow = self
        self.train_wf.gds[0].need_err_input = True
        self.train_wf.link_from(self.start_point)
        self.end_point.link_from(self.train_wf)

    def forward(self, end_unit):
        self.info("Running Forward Propagation...")
        wf = self.train_wf

        wf.stopped = False

        for f in wf.forwards:
            f.gate_block <<= False

        wf.forwards[0].ignores_gate <<= True

        wf.start_point.unlink_after()
        wf.forwards[0].link_from(wf.start_point)

        wf.end_point.unlink_before()
        u = wf[end_unit][-1]  # TODO: u = wf[end_unit]
        for ref_dst in u.links_to.keys():
            ref_dst().gate_block = Bool(True)
        wf.end_point.link_from(u)
        wf.end_point.gate_block = Bool(False)

        wf.run_is_blocking = True
        self.run_is_blocking = True

        wf.run()

    def backward(self, start_unit):
        self.info("Running Backward Propagation...")
        wf = self.train_wf

        wf.stopped = False

        for g in wf.gds:
            g.gate_skip <<= False
            g.gate_block <<= False
            g.gpu_weights_update = nothing
            g.gpu_bias_update = nothing

        start_u = wf[start_unit][0]  # TODO: start_u = wf[start_unit]
        start_u.ignores_gate <<= True

        wf.start_point.unlink_after()
        start_u.link_from(wf.start_point)

        wf.end_point.unlink_before()
        u = wf.gds[0]
        for ref_dst in u.links_to.keys():
            ref_dst().gate_block = Bool(True)
        wf.end_point.link_from(u)
        wf.end_point.gate_block = Bool(False)

        wf.lr_adjuster.gate_block = Bool(True)
        # TODO: not lr_adjuster but next unit after gds[0]

        wf.run_is_blocking = True
        self.run_is_blocking = True

        wf.run()