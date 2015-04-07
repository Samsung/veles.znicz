# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Apr 7, 2015

Prints the table of the label outcomes.

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

import numpy
from zope.interface import implementer

from veles.external.prettytable import PrettyTable
from veles.units import Unit, IUnit


@implementer(IUnit)
class LabelsPrinter(Unit):
    def __init__(self, workflow, **kwargs):
        super(LabelsPrinter, self).__init__(workflow, **kwargs)
        self.top_number = kwargs.get("top_number", 5)
        self.demand("input", "labels_mapping")

    def initialize(self, **kwargs):
        pass

    def run(self):
        self.input.map_read()
        mem = self.input.mem[0]
        labels = [(v, i) for i, v in enumerate(mem)]
        labels.sort(reverse=True)
        table = PrettyTable("label", "value")
        table.float_format = ".5"
        for row in labels[:self.top_number]:
            table.add_row(*reversed(row))
        self.info("Results:\n%s", table)
        self.info("Max to mean ratio: %.1f", numpy.max(mem) / numpy.mean(mem))
