# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Apr 20, 2015

Collects sequential differences and absolute values of registered
:class:`veles.memory.Vector` instances.

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

from collections import defaultdict
import numpy
from zope.interface import implementer

from veles.distributable import IDistributable
from veles.pickle2 import pickle, best_protocol
from veles.units import Unit, IUnit


@implementer(IUnit, IDistributable)
class DiffStats(Unit):
    def __init__(self, workflow, **kwargs):
        view_group = kwargs.get("view_group", "PLOTTER")
        kwargs["view_group"] = view_group
        self._arrays = {u: set(v) for u, v
                        in kwargs.get("arrays", {}).items()}
        super(DiffStats, self).__init__(workflow, **kwargs)
        self._stats = {u: defaultdict(list) for u in self._arrays}
        self._file_name = kwargs.get("file_name")

    def init_unpickled(self):
        super(DiffStats, self).init_unpickled()
        self._previous_ = {u: {} for u in self._arrays}

    @property
    def stats(self):
        return self._stats

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        if not isinstance(value, str):
            raise TypeError("file_name must be a string")
        self._file_name = value

    @property
    def size(self):
        return sum(sum(len(a) for a in v.values())
                   for v in self.stats.values())

    def initialize(self, **kwargs):
        pass

    def run(self):
        for unit, anames in self._arrays.items():
            for aname in anames:
                vector = getattr(unit, aname)
                if not vector:
                    continue
                vector.map_read()
                array = vector.mem
                prev = self._previous_[unit]
                if aname not in prev:
                    prev[aname] = array.copy()
                    continue
                delta = numpy.sum(numpy.abs(array - prev[aname]))
                self.stats[unit][aname].append(
                    {"delta": delta, "abs": numpy.sum(numpy.abs(array))})
                prev[aname][:] = array

    def stop(self):
        self.info("Writing %d items to %s...", self.size, self.file_name)
        with open(self.file_name, "wb") as fout:
            pickle.dump({u.name: vals for u, vals in self.stats.items()}, fout,
                        protocol=best_protocol)

    def register(self, unit, attr):
        self._arrays[unit].add(attr)

    def unregister(self, unit, attr):
        self._arrays[unit].remove(attr)

    def generate_data_for_master(self):
        try:
            return {u: {attr: stats[-1] for attr, stats in vals}
                    for u, vals in self.stats}
        except IndexError:
            return {}

    def generate_data_for_slave(self):
        return None

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        for u, stats in data.items():
            for attr, val in stats.items():
                self.stats[u][attr].append(val)
