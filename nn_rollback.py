# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jul 10, 2014

Unit, whick returns workflow to the saved state, if Model starts to diverge.

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
from veles.units import IUnit, Unit
from veles.distributable import IDistributable


@implementer(IUnit, IDistributable)
class NNRollback(Unit):
    """
    Unit, whick returns workflow to the save state, if Model starts to diverge.
    """
    weights_names = (
        "weights", "bias", "gradient_weights", "gradient_bias")

    def __init__(self, workflow, **kwargs):
        super(NNRollback, self).__init__(workflow, **kwargs)
        self.lr_plus = kwargs.get("lr_plus", 1.04)
        self.lr_minus = kwargs.get("lr_minus", 0.65)
        self.plus_steps = kwargs.get("plus_steps", 1)
        self.minus_steps = kwargs.get("minus_steps", 3)
        self._plus_steps = self.plus_steps
        self._minus_steps = self.minus_steps
        self.improved = None
        self.demand("improved")
        self._gds = {}
        self.history_limit = 2

        # Workaround for difference in minibatch class serve order
        # in clear run and after the resuming from the snapshot.
        self._first_run = True

    def init_unpickled(self):
        super(NNRollback, self).init_unpickled()
        self.slaves = {}

    def initialize(self, **kwargs):
        self.info("lr_plus=%.2f lr_minus=%.2f", self.lr_plus, self.lr_minus)

    def generate_data_for_slave(self, slave):
        self.slaves[slave.id] = 1

    def generate_data_for_master(self):
        return True

    def apply_data_from_master(self, data):
        pass

    def apply_data_from_slave(self, data, slave):
        self._slave_ended(slave)

    def _slave_ended(self, slave):
        if slave.id in self.slaves:
            del self.slaves[slave.id]
        if (not len(self.slaves) and not bool(self.gate_skip)
                and not bool(self.gate_block)):
            self.run()

    def drop_slave(self, slave):
        self._slave_ended(slave)

    def get_weights(self, gd, name, value):
        weights = getattr(gd, name)
        weights.map_read()
        ww = value.get(name, [])
        ww.append(weights.mem.copy())
        while len(ww) > self.history_limit:
            ww.pop(0)
        return ww

    def calculate_nans(self, gd, name):
        weights = getattr(gd, name)
        if weights:
            weights.map_read()
            return numpy.count_nonzero(numpy.isnan(weights.mem))
        else:
            return 0

    def rollback_weights(self, gd, name, value, rollback_to):
        weights = getattr(gd, name)
        ww = value.get(name)
        if ww is None:
            self.warning("No rollback for %s" % name)
        else:
            self.info("Rolling back to stored weights")
            weights.map_invalidate()
            weights_to_return = ww[rollback_to]
            if rollback_to >= 0:
                del ww[rollback_to + 1:]
            return weights_to_return

    def run(self):
        if self.improved:
            self._plus_steps += 1
            if self._plus_steps < self.plus_steps:
                return
            self._plus_steps = 0
            self._minus_steps = 0
            for _gd, kv in self._gds.items():
                k = kv["lr_plus"]
                if k is None:
                    k = self.lr_plus
                _gd.learning_rate *= k
                _gd.learning_rate_bias *= k
                self.info("Increased lr of %s by %.2f, new_lr %.2e",
                          repr(_gd), k, _gd.learning_rate)
                for weights_name in self.weights_names:
                    if getattr(_gd, weights_name, None):
                        kv[weights_name] = self.get_weights(
                            _gd, weights_name, kv)
        elif not self._first_run:
            rollback_to = 0  # -1

            # Check for NaNs
            for _gd, kv in self._gds.items():
                nz = 0
                for weights_name in self.weights_names:
                    nz += self.calculate_nans(_gd, weights_name)
                if nz:
                    self.warning("NaNs encountered, will rollback to -%d",
                                 self.history_limit)
                    self._minus_steps = self.minus_steps
                    rollback_to = 0
                    break

            self._minus_steps += 1
            if self._minus_steps < self.minus_steps:
                return

            self._minus_steps = 0
            self._plus_steps = 0
            for _gd, kv in self._gds.items():
                k = kv["lr_minus"]
                if k is None:
                    k = self.lr_minus
                _gd.learning_rate *= k
                _gd.learning_rate_bias *= k
                self.info("Decreased lr of %s by %.2f, new_lr %.2e",
                          repr(_gd), k, _gd.learning_rate)
                for weights_name in self.weights_names:
                    if getattr(_gd, weights_name, None):
                        setattr(_gd, "%s.mem[:]" % weights_name,
                                self.rollback_weights(
                                    _gd, weights_name, kv, rollback_to))

        self._first_run = False

    def reset(self):
        self._gds.clear()

    def add_gd(self, _gd, lr_plus=None, lr_minus=None):
        kv = self._gds.get(_gd, {})
        kv["lr_plus"] = lr_plus
        kv["lr_minus"] = lr_minus
        self._gds[_gd] = kv
