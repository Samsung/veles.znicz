# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 15, 2013

Decision unit.

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


from __future__ import division
import time

import numpy
import six
from zope.interface import implementer, Interface

from veles.config import root
from veles.distributable import IDistributable
from veles.mutable import Bool
from veles.units import Unit, IUnit
from veles.workflow import NoMoreJobs
from veles.loader import CLASS_NAME, TRAIN, VALID, TEST
from veles.result_provider import IResultProvider
from veles.unit_registry import MappedUnitRegistry


def nvl(x, none_vle):
    return none_vle if x is None else x


def nmax(x, y, none_vle=None):
    return none_vle if x is None and y is None else max(nvl(x, y), nvl(y, x))


def pt_str(x, percent_sign=True):
    return "None" if x is None else ("%.2f%%" if percent_sign else "%.2f") % x


def rpt_str(x):
    return "None" if x is None else "%.2f%%" % (100.0 - x)


class DecisionsRegistry(MappedUnitRegistry):
    mapping = "decisions"
    base = Unit
    loss_mapping = {}

    def __init__(cls, name, bases, clsdict):
        super(DecisionsRegistry, cls).__init__(name, bases, clsdict)
        if ("LOSS" in clsdict and "MAPPING" in clsdict):
            DecisionsRegistry.loss_mapping[clsdict[
                "LOSS"]] = clsdict["MAPPING"]


class IDecision(Interface):
    def on_run():
        """This method is supposed to be overriden in inherited classes.
        """

    def on_last_minibatch():
        """This method is supposed to be overriden in inherited classes.
        """

    def improve_condition():
        """This method is supposed to be overriden in inherited classes.
        """

    def on_training_finished():
        """This method is supposed to be overriden in inherited classes.
        """

    def on_generate_data_for_slave(data):
        """This method is supposed to be overriden in inherited classes.
        """

    def on_generate_data_for_master(data):
        """This method is supposed to be overriden in inherited classes.
        """

    def on_apply_data_from_master(data):
        """This method is supposed to be overriden in inherited classes.
        """

    def on_apply_data_from_slave(data, slave):
        """This method is supposed to be overriden in inherited classes.
        """

    def fill_statistics(stats):
        """This method is supposed to be overriden in inherited classes.
        """

    def fill_snapshot_suffixes(suffixes):
        """This method is supposed to be overriden in inherited classes.
        """

    def stop_condition():
        """This method is supposed to be overriden in inherited classes.
        """


@six.add_metaclass(DecisionsRegistry)
@implementer(IUnit, IDistributable)
class DecisionBase(Unit):
    hide_from_registry = True
    """
    Base class for epoch decision units. Keeps track of learning epochs,
    that is, dataset passes.

    Attributes:
        complete (mutable.Bool): everything's over flag
        improved (mutable.Bool): indicates whether the previous
            epoch's validation results are better than those
            of the epoch before it.
        train_improved (mutable.Bool): like "improved", but for train.
        snapshot_suffix: the suitable suffix for the snapshot file name.

        minibatch_class: from loader (must be set before initialize()!)
        last_minibatch: from loader (must be set before initialize()!)
        class_lengths: from loader (must be set before initialize()!)
        epoch_number: from loader (must be set before initialize()!)
        epoch_ended: from loader (must be set before initialize()!)

    Attributes:
        max_epochs - max number of epochs for training (stop if exceeded)
    """
    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = kwargs.get("view_group", "TRAINER")
        self.complete = Bool(False)
        super(DecisionBase, self).__init__(workflow, **kwargs)
        self.verify_interface(IDecision)
        self.max_epochs = kwargs.get("max_epochs", None)
        self.improved = Bool(False)
        self.improved_epoch_number = None
        self.train_improved = Bool(False)
        self.snapshot_suffix = ""
        self.epoch_timestamp = False
        self.demand("last_minibatch", "minibatch_class",
                    "class_lengths", "epoch_number", "epoch_ended")

    def init_unpickled(self):
        super(DecisionBase, self).init_unpickled()
        self.epoch_timestamp = False

        def on_completed(_):
            self.debug("complete becomes True")

        self.complete.on_true = on_completed

    @property
    def max_epochs(self):
        return self._max_epochs

    @max_epochs.setter
    def max_epochs(self, value):
        if value is None:
            self._max_epochs = None
            return
        if not isinstance(value, int):
            raise TypeError(
                "max_epochs must be an integer or None (got %s)" % type(value))
        if value < 1:
            raise ValueError(
                "max_epochs must be greater than 0 (got %d)" % value)
        self._max_epochs = value

    def initialize(self, **kwargs):
        if self.max_epochs is not None:
            self.info("Will allow max %d epochs", self.max_epochs)
        if self.testing:
            self.improved <<= False
            self.train_improved <<= False
            self.complete <<= False

    def run(self):
        if self.epoch_timestamp is False:
            self.epoch_timestamp = time.time()
        self.on_run()
        if self.is_slave:
            self.complete <<= True
            self.on_last_minibatch()
            self._print_statistics()
        elif self.last_minibatch:
            self._on_last_minibatch()

    def generate_data_for_master(self):
        data = {}
        self.on_generate_data_for_master(data)
        return data

    def generate_data_for_slave(self, slave):
        if self.complete:
            raise NoMoreJobs()
        if self.epoch_timestamp is False:
            self.epoch_timestamp = time.time()
        data = {}
        self.on_generate_data_for_slave(data)
        return data

    def apply_data_from_master(self, data):
        self.complete <<= False
        self.on_apply_data_from_master(data)

    def apply_data_from_slave(self, data, slave):
        if slave is None:
            # Partial update
            return
        self.on_apply_data_from_slave(data, slave)
        if self.last_minibatch:
            self._on_last_minibatch()
        self.has_data_for_slave = not self.complete

    def drop_slave(self, slave):
        pass

    def initialize_arrays(self, minibatch_array, arrays):
        if minibatch_array:
            for index, item in enumerate(arrays):
                if item is None or item.size != len(minibatch_array):
                    arrays[index] = numpy.zeros_like(minibatch_array.mem)
                else:
                    arrays[index][:] = 0
        else:
            import traceback
            stack = traceback.format_stack(limit=2)[:-1]
            self.debug("Did not initialize arrays:\n%s", "\n".join(stack))

    def _on_last_minibatch(self):
        self.on_last_minibatch()

        # Test and Validation sets processed
        if self.epoch_ended:
            self.train_improved <<= self.train_improve_condition()
            improved = self.improve_condition()
            if improved:
                self.improved_epoch_number = self.epoch_number
            self.improved <<= improved
            suffixes = []
            self.fill_snapshot_suffixes(suffixes)
            self.snapshot_suffix = '_'.join(suffixes)
            self.complete <<= self._stop_condition()

        # Training set processed
        if self.minibatch_class == TRAIN:
            self.on_training_finished()

        self._print_statistics()

    def _stop_condition(self):
        if self.testing:
            return True
        # stop if max epoch number was reached or earlier
        return self.stop_condition() or (self.max_epochs is not None and
                                         self.epoch_number >= self.max_epochs)

    def _print_statistics(self):
        stats = []
        self.fill_statistics(stats)
        timestamp = time.time()
        self.info("Epoch %d class %s %s in %.2f sec" %
                  (self.epoch_number, CLASS_NAME[self.minibatch_class],
                   " ".join(stats),
                   timestamp - self.epoch_timestamp))
        self.epoch_timestamp = timestamp


@implementer(IDecision)
class TrivialDecision(DecisionBase):
    def on_run(self):
        pass

    def on_last_minibatch(self):
        pass

    def improve_condition(self):
        return False

    def train_improve_condition(self):
        return False

    def on_training_finished(self):
        pass

    def on_generate_data_for_slave(self, data):
        return None

    def on_generate_data_for_master(self, data):
        return None

    def on_apply_data_from_master(self, data):
        pass

    def on_apply_data_from_slave(self, data, slave):
        pass

    def fill_statistics(self, stats):
        pass

    def fill_snapshot_suffixes(self, suffixes):
        pass

    def stop_condition(self):
        return False


@implementer(IDecision, IResultProvider)
class DecisionGD(DecisionBase):

    MAPPING = "decision_gd"
    LOSS = "softmax"

    """Rules the gradient descent learning process.

    Attributes:
        gd_skip: skip gradient descent or not.
        minibatch_n_err: number of errors for a minibatch.
        epoch_n_err: number of errors for an epoch.
        epoch_n_err_pt: number of errors for an epoch in percents.
        fail_iterations: number of consequent iterations with non-decreased
                         validation error.
        confusion_matrixes: confusion matrixes.
        minibatch_confusion_matrix: confusion matrix for a minibatch.
        minibatch_max_err_y_sum: maximum of backpropagated gradient
                                 for a minibatch.
        max_err_y_sums: maximums of backpropagated gradient.
    """
    BIGNUM = 1.0e30

    def __init__(self, workflow, **kwargs):
        super(DecisionGD, self).__init__(workflow, **kwargs)
        self.fail_iterations = kwargs.get("fail_iterations", 100)
        self.gd_skip = Bool()

        # Values for the current epoch
        self.epoch_n_err = [None] * 3
        self.epoch_n_evaluated_samples = [0] * 3
        self.epoch_n_err_pt = [None] * 3

        # Best achieved errors, independently for each class
        self.best_n_err_pt = [None] * 3
        # and its epoch numbers
        self.best_n_err_pt_epoch_number = [None] * 3
        # errors for other classes when the given class was the best
        self.best_n_err_pt_others = [[None] * 3] * 3
        self._store_best_n_err_pt_others = [False] * 3

        # Errors at the epoch where
        # max of train and validation errors was the best
        self.best_minimax_n_err_pt = [None] * 3
        # and it's epoch number
        self.best_minimax_n_err_pt_epoch_number = -1

        self.minibatch_n_err = None  # memory.Array()
        self.minibatch_confusion_matrix = None  # memory.Array()
        self.minibatch_max_err_y_sum = None  # memory.Array()

        self.confusion_matrixes = [None] * 3
        self.max_err_y_sums = [0] * 3

        self.autoencoder = False

        self.demand("minibatch_size")

    def initialize(self, **kwargs):
        super(DecisionGD, self).initialize(**kwargs)
        # Reset errors
        self.epoch_n_err[:] = [None] * 3
        self.epoch_n_evaluated_samples[:] = [0] * 3
        self.epoch_n_err_pt[:] = [None] * 3
        map(self.reset_statistics, range(3))
        self.initialize_arrays(self.minibatch_confusion_matrix,
                               self.confusion_matrixes)

    def get_metric_names(self):
        if not self.testing:
            return {"Min errors", "Accuracy", "EvaluationFitness",
                    "Best epoch"}
        return set()

    def get_metric_values(self):
        if self.testing:
            return {}
        tstr = CLASS_NAME[TRAIN]
        vstr = CLASS_NAME[VALID]
        cstr = "minimax(%s, %s)" % (tstr, vstr)
        evalfun = root.common.evaluation_transform
        return {
            "Min errors": {
                tstr: pt_str(self.best_n_err_pt[TRAIN]),
                vstr: pt_str(self.best_n_err_pt[VALID]),
                cstr: pt_str(
                    nmax(self.best_minimax_n_err_pt[VALID],
                         self.best_minimax_n_err_pt[TRAIN]))
            },
            "Accuracy": {
                tstr: rpt_str(self.best_n_err_pt[TRAIN]),
                vstr: rpt_str(self.best_n_err_pt[VALID]),
                cstr: rpt_str(
                    nmax(self.best_minimax_n_err_pt[VALID],
                         self.best_minimax_n_err_pt[TRAIN]))
            },
            "EvaluationFitness": evalfun(
                1 - nvl(self.best_n_err_pt[VALID], 100) / 100,
                1 - nvl(self.best_n_err_pt[TRAIN], 100) / 100),
            "Best epoch": {
                tstr: nvl(self.best_n_err_pt_epoch_number[TRAIN], "None"),
                vstr: nvl(self.best_n_err_pt_epoch_number[VALID], "None"),
                cstr: nvl(self.best_minimax_n_err_pt_epoch_number,
                          "None")
            }}

    def on_run(self):
        # Check skip gradient descent or not
        self.gd_skip <<= (self.minibatch_class != TRAIN)

    def on_last_minibatch(self):
        minibatch_class = self.minibatch_class
        # Copy confusion matrix
        if (self.minibatch_confusion_matrix is not None and
                self.minibatch_confusion_matrix.mem is not None):
            self.minibatch_confusion_matrix.map_read()
            self.confusion_matrixes[minibatch_class][:] = (
                self.minibatch_confusion_matrix.mem[:])

        if self.minibatch_n_err:
            self.minibatch_n_err.map_read()
            self.epoch_n_err[minibatch_class] = self.minibatch_n_err[0]
            self.epoch_n_evaluated_samples[minibatch_class] = (
                self.minibatch_n_err[1])
            # Calculate error in percent
            if self.class_lengths[minibatch_class]:
                self.epoch_n_err_pt[minibatch_class] = (
                    100.0 * self.epoch_n_err[minibatch_class] /
                    self.epoch_n_evaluated_samples[minibatch_class])
                # Update best error
                if (self.epoch_n_err_pt[minibatch_class] <
                        nvl(self.best_n_err_pt[minibatch_class], self.BIGNUM)):
                    self.best_n_err_pt[minibatch_class] = (
                        self.epoch_n_err_pt[minibatch_class])
                    self.best_n_err_pt_epoch_number[minibatch_class] = (
                        self.epoch_number)
                    self._store_best_n_err_pt_others[minibatch_class] = True

        # Store maximum of backpropagated gradient
        if (self.minibatch_max_err_y_sum is not None and
                self.minibatch_max_err_y_sum.mem is not None):
            self.minibatch_max_err_y_sum.map_read()
            self.max_err_y_sums[minibatch_class] = (
                self.minibatch_max_err_y_sum[0])

    def improve_condition(self):
        """Called at the end of an epoch.

        minibatch_class will be VALID if validation exists, else TRAIN.
        """
        for i, store in enumerate(self._store_best_n_err_pt_others):
            if store:
                self.best_n_err_pt_others[i][:] = self.epoch_n_err_pt
                self._store_best_n_err_pt_others[i] = False

        minibatch_class = self.minibatch_class
        if (nmax(self.epoch_n_err_pt[minibatch_class],
                 self.epoch_n_err_pt[TRAIN], self.BIGNUM) <
            nmax(self.best_minimax_n_err_pt[minibatch_class],
                 self.best_minimax_n_err_pt[TRAIN], self.BIGNUM)):
            for i in (minibatch_class, TRAIN, TEST):
                self.best_minimax_n_err_pt[i] = self.epoch_n_err_pt[i]
            self.best_minimax_n_err_pt_epoch_number = self.epoch_number
            return True
        return False

    def train_improve_condition(self):
        if (nvl(self.epoch_n_err_pt[TRAIN], self.BIGNUM) <
                nvl(self.best_n_err_pt[TRAIN], self.BIGNUM)):
            self.best_n_err_pt[TRAIN] = self.epoch_n_err_pt[TRAIN]
            self.best_n_err_pt_epoch_number[TRAIN] = self.epoch_number
            self._store_best_n_err_pt_others[TRAIN] = True
            return True
        return False

    def on_training_finished(self):
        pass

    def on_generate_data_for_master(self, data):
        for attr in ["minibatch_n_err", "minibatch_max_err_y_sum",
                     "minibatch_confusion_matrix"]:
            attrval = getattr(self, attr)
            if attrval is not None:
                attrval.map_read()
                data[attr] = attrval.mem

    def on_generate_data_for_slave(self, data):
        data["improved"] = bool(self.improved)

    def on_apply_data_from_master(self, data):
        self.improved <<= data["improved"]
        self.reset_statistics(self.minibatch_class)
        # To stop just after the first minibatch
        self.best_minimax_n_err_pt[VALID] = 0
        self.best_minimax_n_err_pt[TRAIN] = 0

    def on_apply_data_from_slave(self, data, slave):
        if self.minibatch_n_err:
            self.minibatch_n_err.map_write()
            self.minibatch_n_err.mem += data["minibatch_n_err"]
        if self.minibatch_max_err_y_sum is not None:
            self.minibatch_max_err_y_sum.map_write()
            numpy.maximum(self.minibatch_max_err_y_sum.mem,
                          data["minibatch_max_err_y_sum"],
                          self.minibatch_max_err_y_sum.mem)
        if self.minibatch_confusion_matrix is not None:
            self.minibatch_confusion_matrix.map_write()
            self.minibatch_confusion_matrix.mem += data[
                "minibatch_confusion_matrix"]

    def stop_condition(self):
        if all(nvl(self.best_minimax_n_err_pt[i], 0) <= 0
               for i in (VALID, TRAIN)):
            return True
        if (self.epoch_number - self.improved_epoch_number >
                self.fail_iterations):
            return True
        return False

    def fill_statistics(self, ss):
        minibatch_class = self.minibatch_class
        if self.minibatch_n_err is not None and not self.autoencoder:
            if (self.epoch_n_err[minibatch_class] == 0 and
                    self.epoch_number == 0):
                self.warning("Number of errors equals to 0 before the training"
                             " has actually started => dropping into pdb...")
                import pdb
                pdb.set_trace()
            ss.append("n_err %d of %d (%.2f%%)" %
                      (self.epoch_n_err[minibatch_class],
                       self.epoch_n_evaluated_samples[minibatch_class],
                       self.epoch_n_err_pt[minibatch_class]))
        if not self.is_slave:  # we will need them in generate_data_for_master
            self.reset_statistics(self.minibatch_class)

    def fill_snapshot_suffixes(self, ss):
        if self.minibatch_n_err is not None:
            for set_samples in(TEST, VALID, TRAIN):
                if self.epoch_n_err_pt[set_samples] is not None:
                    ss.append(
                        "%s_%s" % (
                            CLASS_NAME[set_samples],
                            pt_str(self.epoch_n_err_pt[set_samples], False)))

    def reset_statistics(self, minibatch_class):
        # Reset statistics per class
        for vec in (self.minibatch_n_err, self.minibatch_max_err_y_sum,
                    self.minibatch_confusion_matrix):
            if not vec:
                continue
            vec.map_invalidate()
            vec.mem[:] = 0


class DecisionMSE(DecisionGD):

    MAPPING = "decision_mse"
    LOSS = "mse"

    """Rules the gradient descent mean square error (MSE) learning process.

    Attributes:
        epoch_metrics: metrics for an epoch (same as minibatch_metrics).
    """
    def __init__(self, workflow, **kwargs):
        super(DecisionMSE, self).__init__(workflow, **kwargs)

        # Values for the current epoch
        self.epoch_mse = [None] * 3

        # Best achieved MSE, independently for each class
        self.best_mse = [None] * 3
        # and its epoch numbers
        self.best_mse_epoch_number = [None] * 3
        # MSE for other classes when the given class was the best
        self.best_mse_others = [[None] * 3] * 3
        self._store_best_mse_others = [False] * 3

        # MSE at the epoch where
        # max of train and validation MSE was the best
        self.best_minimax_mse = [None] * 3
        # and it's epoch number
        self.best_minimax_mse_epoch_number = -1

        self.epoch_metrics = [None] * 3
        self.root = kwargs.get("root", True)
        self.demand("minibatch_metrics", "minibatch_class", "class_lengths")

    def initialize(self, **kwargs):
        super(DecisionMSE, self).initialize(**kwargs)
        self.initialize_arrays(self.minibatch_metrics, self.epoch_metrics)

    def get_metric_names(self):
        if self.testing:
            return set()
        names = super(DecisionMSE, self).get_metric_names()
        mstr = "RMSE" if self.root else "MSE"
        tstr = CLASS_NAME[TRAIN]
        vstr = CLASS_NAME[VALID]
        names.update({mstr, "Min %s epochs number" % mstr,
                      "%s %s on min %s %s" % (tstr, mstr, vstr, mstr),
                      "EvaluationFitness"})
        return names

    def get_metric_values(self):
        if self.testing:
            return {}
        values = super(DecisionMSE, self).get_metric_values()
        mstr = "RMSE" if self.root else "MSE"
        tstr = CLASS_NAME[TRAIN]
        vstr = CLASS_NAME[VALID]
        cstr = "minimax(%s, %s)" % (tstr, vstr)
        evalfun = root.common.evaluation_transform
        values.update({
            mstr: {tstr: "%.12f" % self.best_mse[TRAIN],
                   vstr: "%.12f" % self.best_mse[VALID],
                   cstr: "%.12f" %
                   nmax(self.best_minimax_mse[VALID],
                        self.best_minimax_mse[TRAIN])},
            "EvaluationFitness":
            evalfun(-self.best_minimax_mse[VALID], -self.best_minimax_mse[
                    TRAIN]),
            "Min %s epochs number" % mstr:
            {tstr: self.best_mse_epoch_number[TRAIN],
             vstr: self.best_mse_epoch_number[VALID],
             cstr: self.best_minimax_mse_epoch_number},
            "%s %s on min %s %s" % (tstr, mstr, vstr, mstr):
            self.best_mse_others[VALID][TRAIN]})
        return values

    def on_last_minibatch(self):
        super(DecisionMSE, self).on_last_minibatch()

        # minibatch_metrics: [SUM((R)MSE), min((R)MSE), max((R)MSE)]

        minibatch_class = self.minibatch_class
        self.minibatch_metrics.map_read()

        # Copy metrics
        self.epoch_metrics[minibatch_class][:] = (
            self.minibatch_metrics.mem[:])

        # Compute average mse
        self.epoch_metrics[minibatch_class][0] = (
            self.epoch_metrics[minibatch_class][0] /
            self.class_lengths[minibatch_class])

        if self.epoch_number == 0:
            self.epoch_metrics[TRAIN][:] = self.epoch_metrics[VALID][:]

    def improve_condition(self):
        if (nvl(self.epoch_metrics[VALID][0], self.BIGNUM) <
                nvl(self.best_mse[VALID], self.BIGNUM)):
            self.best_mse[VALID] = self.epoch_metrics[VALID][0]
            self.best_mse_epoch_number[VALID] = self.epoch_number
            self._store_best_mse_others[VALID] = True

        for i, store in enumerate(self._store_best_mse_others):
            if store:
                self.best_mse_others[i][:] = (x[0] for x in self.epoch_metrics)
                self._store_best_mse_others[i] = False

        minibatch_class = self.minibatch_class
        if (nmax(self.epoch_metrics[minibatch_class][0],
                 self.epoch_metrics[TRAIN][0], self.BIGNUM) <
            nmax(self.best_minimax_mse[minibatch_class],
                 self.best_minimax_mse[TRAIN], self.BIGNUM)):
            for i in (minibatch_class, TRAIN, TEST):
                self.best_minimax_mse[i] = self.epoch_metrics[i][0]
            self.best_minimax_mse_epoch_number = self.epoch_number
            return True

        return super(DecisionMSE, self).improve_condition()

    def train_improve_condition(self):
        if (nvl(self.epoch_metrics[TRAIN][0], self.BIGNUM) <
                nvl(self.best_mse[TRAIN], self.BIGNUM)):
            self.best_mse[TRAIN] = self.epoch_metrics[TRAIN][0]
            self.best_mse_epoch_number[TRAIN] = self.epoch_number
            self._store_best_mse_others[TRAIN] = True
            return True
        return super(DecisionMSE, self).train_improve_condition()

    def on_generate_data_for_master(self, data):
        super(DecisionMSE, self).on_generate_data_for_master(data)
        for attr in ("minibatch_metrics",):
            attrval = getattr(self, attr)
            if attrval is not None:
                attrval.map_read()
                data[attr] = attrval.mem

    def on_apply_data_from_master(self, data):
        super(DecisionMSE, self).on_apply_data_from_master(data)
        # To stop just after the first minibatch
        self.best_minimax_mse[TRAIN] = 0
        self.best_minimax_mse[VALID] = 0

    def on_apply_data_from_slave(self, data, slave):
        super(DecisionMSE, self).on_apply_data_from_slave(data, slave)
        if self.minibatch_metrics is not None:
            self.minibatch_metrics.map_write()
            self.minibatch_metrics[0] += data["minibatch_metrics"][0]
            self.minibatch_metrics[1] = numpy.maximum(
                self.minibatch_metrics[1], data["minibatch_metrics"][1])
            self.minibatch_metrics[2] = numpy.minimum(
                self.minibatch_metrics[2], data["minibatch_metrics"][2])

    def fill_snapshot_suffixes(self, ss):
        if self.minibatch_metrics is not None:
            for mc in VALID, TRAIN:
                ss.append("%.4f" % self.epoch_metrics[mc][0])
        super(DecisionMSE, self).fill_snapshot_suffixes(ss)

    def fill_statistics(self, ss):
        minibatch_class = self.minibatch_class
        if self.epoch_metrics[minibatch_class] is not None:
            ss.append("%s %.6f (max %.6f; min %.3e)" %
                      (("RMSE" if self.root else "MSE",) +
                       tuple(self.epoch_metrics[minibatch_class])))
        super(DecisionMSE, self).fill_statistics(ss)

    def reset_statistics(self, minibatch_class):
        super(DecisionMSE, self).reset_statistics(minibatch_class)
        # Reset statistics per class
        if (self.minibatch_metrics is not None and
                self.minibatch_metrics.mem is not None):
            self.minibatch_metrics.map_invalidate()
            self.minibatch_metrics.mem[:] = 0

    def stop_condition(self):
        if all(nvl(self.best_minimax_mse[i], 0) <= 0 for i in (VALID, TRAIN)):
            return True
        if (self.epoch_number - self.improved_epoch_number >
                self.fail_iterations):
            return True
        return False
