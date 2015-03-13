"""
Created on Aug 15, 2013

DecisionGD unit.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import time

import numpy
from zope.interface import implementer, Interface

from veles.distributable import IDistributable
from veles.mutable import Bool
from veles.units import Unit, IUnit
from veles.workflow import NoMoreJobs
from veles.loader import CLASS_NAME, TRAIN


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


@implementer(IUnit, IDistributable)
class DecisionBase(Unit):
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

    def initialize(self, **kwargs):
        pass

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
            self.info("Did not initialize arrays:\n%s", "\n".join(stack))

    def _on_last_minibatch(self):
        self.on_last_minibatch()

        # Test and Validation sets processed
        if self.epoch_ended:
            self.train_improved <<= self.train_improve_condition()
            self.improved <<= self.improve_condition()
            if self.improved:
                suffixes = []
                self.fill_snapshot_suffixes(suffixes)
                self.snapshot_suffix = '_'.join(suffixes)
            self.complete <<= self._stop_condition()

        # Training set processed
        if self.minibatch_class == TRAIN:
            self.on_training_finished()

        self._print_statistics()

    def _stop_condition(self):
        if self.stop_condition():
            return True
        # stop if max epoch number was reached
        if (self.max_epochs is not None and
                self.epoch_number >= self.max_epochs):
            return True
        return False

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


@implementer(IDecision)
class DecisionGD(DecisionBase):
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
    def __init__(self, workflow, **kwargs):
        super(DecisionGD, self).__init__(workflow, **kwargs)
        self.fail_iterations = kwargs.get("fail_iterations", 100)
        self.gd_skip = Bool(False)
        self.epoch_n_err = [1.0e30, 1.0e30, 1.0e30]
        self.epoch_n_err_pt = [100.0, 100.0, 100.0]
        self.minibatch_n_err = None  # formats.Vector()

        # minimum validation error and its epoch number
        self.min_validation_n_err = 1.0e30
        self.min_validation_n_err_epoch_number = -1

        # train error when validation was minimum last time
        self.min_train_validation_n_err = 1.0e30

        # minimum train error and its epoch number
        self.min_train_n_err = 1.0e30
        self.min_train_n_err_epoch_number = -1

        self.confusion_matrixes = [None, None, None]
        self.minibatch_confusion_matrix = None  # formats.Vector()
        self.max_err_y_sums = [0, 0, 0]
        self.minibatch_max_err_y_sum = None  # formats.Vector()
        self.prev_train_err = 1.0e30
        self.demand("minibatch_size")

    def initialize(self, **kwargs):
        super(DecisionGD, self).initialize(**kwargs)
        # Reset errors
        self.epoch_n_err[:] = [1.0e30, 1.0e30, 1.0e30]
        self.epoch_n_err_pt[:] = [100.0, 100.0, 100.0]
        map(self.reset_statistics, range(3))
        self.initialize_arrays(self.minibatch_confusion_matrix,
                               self.confusion_matrixes)

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
            # Calculate error in percent
            if self.class_lengths[minibatch_class]:
                self.epoch_n_err_pt[minibatch_class] = (
                    100.0 * self.epoch_n_err[minibatch_class] /
                    (self.class_lengths[minibatch_class] if not self.is_slave
                     else self.minibatch_size))

        # Store maximum of backpropagated gradient
        if (self.minibatch_max_err_y_sum is not None and
                self.minibatch_max_err_y_sum.mem is not None):
            self.minibatch_max_err_y_sum.map_read()
            self.max_err_y_sums[minibatch_class] = (
                self.minibatch_max_err_y_sum[0])

    def improve_condition(self):
        minibatch_class = self.minibatch_class
        if ((self.epoch_n_err[minibatch_class] < self.min_validation_n_err or
             (self.epoch_n_err[minibatch_class] == self.min_validation_n_err
              and self.epoch_n_err[TRAIN] < self.min_train_validation_n_err))):
            self.min_validation_n_err = self.epoch_n_err[minibatch_class]
            self.min_train_validation_n_err = self.epoch_n_err[TRAIN]
            self.min_validation_n_err_epoch_number = self.epoch_number
            self.workflow.fitness = (
                100.0 - 100.0 * self.epoch_n_err[minibatch_class] /
                (self.class_lengths[minibatch_class] if not self.is_slave
                 else self.minibatch_size))
            return True
        return False

    def train_improve_condition(self):
        if self.epoch_n_err[TRAIN] < self.min_train_n_err:
            self.min_train_n_err = self.epoch_n_err[TRAIN]
            self.min_train_n_err_epoch_number = self.epoch_number
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
        self.min_validation_n_err = 0
        self.min_train_n_err = 0

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
        if self.min_validation_n_err <= 0:
            return True
        if (self.epoch_number - self.min_validation_n_err_epoch_number >
                self.fail_iterations):
            return True
        return False

    def fill_statistics(self, ss):
        minibatch_class = self.minibatch_class
        if self.minibatch_n_err is not None:
            if (self.epoch_n_err[minibatch_class] == 0 and
                    self.epoch_number == 0):
                self.warning("Number of errors equals to 0 before the training"
                             " has actually started")
            ss.append("n_err %d (%.2f%%)" %
                      (self.epoch_n_err[minibatch_class],
                       self.epoch_n_err_pt[minibatch_class]))
        if not self.is_slave:  # we will need them in generate_data_for_master
            self.reset_statistics(self.minibatch_class)

    def fill_snapshot_suffixes(self, ss):
        if self.minibatch_n_err is not None:
            ss.append("%.2fpt" % (self.epoch_n_err_pt[self.minibatch_class]))

    def reset_statistics(self, minibatch_class):
        # Reset statistics per class
        for vec in (self.minibatch_n_err, self.minibatch_max_err_y_sum,
                    self.minibatch_confusion_matrix):
            if not vec:
                continue
            vec.map_invalidate()
            vec.mem[:] = 0


class DecisionMSE(DecisionGD):
    """Rules the gradient descent mean square error (MSE) learning process.

    Attributes:
        epoch_min_mse: minimum mse by class for one epoch.
        epoch_metrics: metrics for an epoch (same as minibatch_metrics).
    """
    def __init__(self, workflow, **kwargs):
        super(DecisionMSE, self).__init__(workflow, **kwargs)
        self.epoch_min_mse = [1.0e30, 1.0e30, 1.0e30]
        self.min_validation_mse = 1.0e30
        self.min_validation_mse_epoch_number = -1
        self.min_train_validation_mse = 1.0e30
        self.min_train_mse = 1.0e30
        self.min_train_mse_epoch_number = -1
        self.epoch_metrics = [None, None, None]
        self.minibatch_metrics = None
        self.demand("minibatch_metrics")

    def initialize(self, **kwargs):
        super(DecisionMSE, self).initialize(**kwargs)
        self.epoch_min_mse[:] = [1.0e30, 1.0e30, 1.0e30]
        self.initialize_arrays(self.minibatch_metrics, self.epoch_metrics)

    def on_last_minibatch(self):
        super(DecisionMSE, self).on_last_minibatch()

        minibatch_class = self.minibatch_class
        self.minibatch_metrics.map_read()
        self.epoch_min_mse[minibatch_class] = (
            self.minibatch_metrics[0] / self.class_lengths[minibatch_class])
        # Copy metrics
        self.epoch_metrics[minibatch_class][:] = (
            self.minibatch_metrics.mem[:])
        # Compute average mse
        self.epoch_metrics[minibatch_class][0] = (
            self.epoch_metrics[minibatch_class][0] /
            self.class_lengths[minibatch_class])

    def improve_condition(self):
        minibatch_class = self.minibatch_class
        if (self.epoch_min_mse[minibatch_class] < self.min_validation_mse or
                (self.epoch_min_mse[minibatch_class] == self.min_validation_mse
                 and self.epoch_min_mse[TRAIN] < self.min_train_mse)):
            self.min_validation_mse = self.epoch_min_mse[minibatch_class]
            self.min_validation_mse_epoch_number = self.epoch_number
            self.min_train_validation_mse = self.epoch_min_mse[TRAIN]
            self.workflow.fitness = 1.0 - self.epoch_min_mse[minibatch_class]
            return True
        return super(DecisionMSE, self).improve_condition()

    def train_improve_condition(self):
        if self.epoch_min_mse[TRAIN] < self.min_train_mse:
            self.min_train_mse = self.epoch_min_mse[TRAIN]
            self.min_train_mse_epoch_number = self.epoch_number
            return True
        return super(DecisionMSE, self).train_improve_condition()

    def on_generate_data_for_master(self, data):
        super(DecisionMSE, self).on_generate_data_for_master(data)
        for attr in ["minibatch_metrics"]:
            attrval = getattr(self, attr)
            if attrval is not None:
                attrval.map_read()
                data[attr] = attrval.mem

    def on_apply_data_from_master(self, data):
        super(DecisionMSE, self).on_apply_data_from_master(data)
        # To stop just after the first minibatch
        self.min_validation_mse = 0
        self.min_train_mse = 0

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
            ss.append("%.6f" % (self.epoch_metrics[self.minibatch_class][0]))
        super(DecisionMSE, self).fill_snapshot_suffixes(ss)

    def fill_statistics(self, ss):
        minibatch_class = self.minibatch_class
        if self.epoch_metrics[minibatch_class] is not None:
            ss.append("AvgMSE %.6f MaxMSE %.6f "
                      "MinMSE %.3e" % (self.epoch_metrics[minibatch_class][0],
                                       self.epoch_metrics[minibatch_class][1],
                                       self.epoch_metrics[minibatch_class][2]))
        super(DecisionMSE, self).fill_statistics(ss)

    def reset_statistics(self, minibatch_class):
        super(DecisionMSE, self).reset_statistics(minibatch_class)
        # Reset statistics per class
        if (self.minibatch_metrics is not None and
                self.minibatch_metrics.mem is not None):
            self.minibatch_metrics.map_invalidate()
            self.minibatch_metrics.mem[:] = 0
            self.minibatch_metrics[2] = 1.0e30

    def stop_condition(self):
        if self.min_validation_mse <= 0:
            return True
        if (self.epoch_number - self.min_validation_mse_epoch_number >
                self.fail_iterations):
            return True
        return False
