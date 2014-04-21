"""
Created on Aug 15, 2013

Decision unit.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy
import os
import six
from six.moves import cPickle as pickle
import sys
import time

import veles.config as config
import veles.formats as formats
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.units as units
from veles.znicz.loader import CLASS_NAME, TRAIN, VALID, TEST


if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    FileNotFoundError = IOError  # pylint: disable=W0622


class Decision(units.Unit):
    """Decides on the learning behavior.

    Attributes:
        complete: completed.
        minibatch_class: current minibatch class.
        no_more_minibatches_left: if current minibatch is last in it's class.
        gd_skip: skip gradient descent or not.
        epoch_number: epoch number.
        epoch_ended: if an epoch has just ended.
        epoch_min_mse: minimum mse by class per epoch.
        minibatch_n_err: number of errors for a minibatch.
        epoch_n_err: number of errors for an epoch.
        epoch_n_err_pt: number of errors for an epoch in percents.
        minibatch_metrics: [0] - mse, [1] - max of sum of sample graidents.
        class_samples: number of samples per class.
        fail_iterations: number of consequent iterations with non-decreased
                         validation error.
        max_epochs: max number of epochs for training (stop if exceeded).
        epoch_metrics: metrics for an epoch (same as minibatch_metrics).
        workflow: reference to workflow to snapshot.
        snapshot_prefix: prefix for the snapshots.
        fnme: filename of the last snapshot.
        fnmeWb: filename of the last weights + bias snapshot.
        just_snapshotted: True after snapshot.
        snapshot_time: time of the last snapshot.
        confusion_matrixes: confusion matrixes.
        minibatch_confusion_matrix: confusion matrix for a minibatch.
        epoch_samples_mse: mse for each sample in the previous epoch
                           (can be used for Histogram plotter).
        tmp_epoch_samples_mse: mse for each sample in the current epoch.
        minibatch_max_err_y_sum: maximum of backpropagated gradient
                                 for a minibatch.
        max_err_y_sums: maximums of backpropagated gradient.
        vectors_to_sync: list of Vector() objects to sync after each epoch.
        sample_input: will be a copy of first element from forward[0].input
                      if the latter is in vectors_to_sync.
        sample_output: will be a copy of first element from forward[-1].output
                       if the latter is in vectors_to_sync.
        sample_target: will be a copy of first element from ev.target
                       if the latter is in vectors_to_sync.
        sample_label: will be a copy of first element label from ev.labels
                       if the latter is in vectors_to_sync.
        use_dynamic_alpha: will adjust alpha according to previous train error.
    """
    def __init__(self, workflow, **kwargs):
        fail_iterations = kwargs.get("fail_iterations", 100)
        snapshot_prefix = kwargs.get("snapshot_prefix", "")
        store_samples_mse = kwargs.get("store_samples_mse", False)
        use_dynamic_alpha = kwargs.get("use_dynamic_alpha", False)
        max_epochs = kwargs.get("max_epochs", None)
        kwargs["fail_iterations"] = fail_iterations
        kwargs["snapshot_prefix"] = snapshot_prefix
        kwargs["store_samples_mse"] = store_samples_mse
        kwargs["use_dynamic_alpha"] = use_dynamic_alpha
        kwargs["view_group"] = kwargs.get("view_group", "TRAINER")
        kwargs["max_epochs"] = max_epochs
        super(Decision, self).__init__(workflow, **kwargs)
        self.class_samples = None  # [0, 0, 0]
        self.def_attr("fail_iterations", fail_iterations)
        self.complete = Bool(False)
        self.gd_skip = Bool(False)
        self.def_attr("epoch_number", 0)
        self.epoch_ended = Bool(False)
        self.epoch_min_mse = [1.0e30, 1.0e30, 1.0e30]
        self.epoch_n_err = [1.0e30, 1.0e30, 1.0e30]
        self.epoch_n_err_pt = [100.0, 100.0, 100.0]
        self.minibatch_n_err = None  # formats.Vector()
        self.minibatch_metrics = None  # formats.Vector()
        self.min_validation_mse = 1.0e30
        self.min_validation_mse_epoch_number = -1
        self.min_train_mse = 1.0e30
        self.min_validation_n_err = 1.0e30
        self.min_validation_n_err_epoch_number = -1
        self.min_train_n_err = 1.0e30
        self.snapshot_prefix = snapshot_prefix
        self._do_snapshots = kwargs.get("do_snapshots", True)
        self._do_export_weights = kwargs.get("do_export_weights", False)
        self.fnme = None
        self.fnmeWb = None
        self.epoch_metrics = [None, None, None]
        self.just_snapshotted = Bool(False)
        self.def_attr("snapshot_time", 0)
        self.minibatch_mse = None
        self.epoch_samples_mse = ([formats.Vector(),
                                   formats.Vector(),
                                   formats.Vector()]
                                  if store_samples_mse
                                  else [])
        self.tmp_epoch_samples_mse = ([formats.Vector(),
                                       formats.Vector(),
                                       formats.Vector()]
                                      if store_samples_mse
                                      else [])
        self.minibatch_offset = None
        self.minibatch_size = None
        self.confusion_matrixes = [None, None, None]
        self.minibatch_confusion_matrix = None  # formats.Vector()
        self.max_err_y_sums = [0, 0, 0]
        self.minibatch_max_err_y_sum = None  # formats.Vector()
        self.vectors_to_sync = {}
        self.def_attr("sample_input", None)
        self.def_attr("sample_output", None)
        self.def_attr("sample_target", None)
        self.def_attr("sample_label", None)
        self.use_dynamic_alpha = use_dynamic_alpha
        self.prev_train_err = 1.0e30
        self.ev = None
        self.max_epochs = max_epochs  # max epochs to learn

    def init_unpickled(self):
        super(Decision, self).init_unpickled()
        self.minibatches_balance_ = [0, 0, 0]
        self.slave_minibatch_class_ = {}

    def initialize(self):
        super(Decision, self).initialize()
        # Reset errors
        self.epoch_min_mse[:] = [1.0e30, 1.0e30, 1.0e30]
        self.epoch_n_err[:] = [1.0e30, 1.0e30, 1.0e30]
        self.epoch_n_err_pt[:] = [100.0, 100.0, 100.0]
        map(self._reset_statistics, range(3))

        # Allocate arrays for confusion matrixes.
        if (self.minibatch_confusion_matrix is not None and
                self.minibatch_confusion_matrix.v is not None):
            for i in range(len(self.confusion_matrixes)):
                if (self.confusion_matrixes[i] is None or
                        self.confusion_matrixes[i].size !=
                        self.minibatch_confusion_matrix.v.size):
                    self.confusion_matrixes[i] = (
                        numpy.zeros_like(self.minibatch_confusion_matrix.v))
                else:
                    self.confusion_matrixes[i][:] = 0

        # Allocate arrays for epoch metrics.
        if (self.minibatch_metrics is not None and
                self.minibatch_metrics.v is not None):
            for i in range(len(self.epoch_metrics)):
                if (self.epoch_metrics[i] is None or
                        self.epoch_metrics[i].size !=
                        self.minibatch_metrics.v.size):
                    self.epoch_metrics[i] = (
                        numpy.zeros_like(self.minibatch_metrics.v))
                else:
                    self.epoch_metrics[i][:] = 0

        # Allocate vectors for storing samples mse.
        for i in range(len(self.epoch_samples_mse)):
            if self.class_samples[i] <= 0:
                continue
            if (self.tmp_epoch_samples_mse[i].v is None or
                    self.tmp_epoch_samples_mse[i].v.size !=
                    self.class_samples[i]):
                self.tmp_epoch_samples_mse[i].v = (
                    numpy.zeros(
                        self.class_samples[i],
                        dtype=opencl_types.dtypes[config.root.common.dtype]))
                self.epoch_samples_mse[i].v = (
                    numpy.zeros(
                        self.class_samples[i],
                        dtype=opencl_types.dtypes[config.root.common.dtype]))
            else:
                self.tmp_epoch_samples_mse[i].v[:] = 0
                self.epoch_samples_mse[i].v[:] = 0

        # Initialize sample_input, sample_output, sample_target if necessary
        if self.workflow.forward[0].input in self.vectors_to_sync:
            self.sample_input = numpy.zeros_like(
                self.workflow.forward[0].input[0])
        if self.workflow.forward[-1].output in self.vectors_to_sync:
            self.sample_output = numpy.zeros_like(
                self.workflow.forward[-1].output[0])
        ev = self.ev if self.ev is not None else self.workflow.ev
        if (ev.__dict__.get("target") is not None
                and ev.target in self.vectors_to_sync
                and ev.target.v is not None):
            self.sample_target = numpy.zeros_like(ev.target[0])

        timestamp = time.time()
        self.epoch_timestamps = [timestamp, timestamp, timestamp]

        if self.is_slave:
            self._reset_statistics_ = self._reset_statistics
            self._on_training_finished_ = self._on_training_finished
            self._reset_statistics = self.nothing
            self._training_finished = self.nothing

    def run(self):
        self.epoch_ended << False
        minibatch_class = self.minibatch_class

        self._copy_minibatch_mse(minibatch_class, self.minibatch_size,
                                 self.minibatch_offset)

        # Check skip gradient descent or not
        self.gd_skip << (minibatch_class != TRAIN)

        if self.no_more_minibatches_left[minibatch_class]:
            self._on_last_minibatch(minibatch_class)

    def _on_snapshot(self, minibatch_class):
        to_rm = []
        if self.fnme is not None:
            to_rm.append("%s.bak" % (self.fnme))
            try:
                os.unlink(to_rm[-1])
            except OSError:
                pass
            try:
                os.rename(self.fnme, to_rm[-1])
            except OSError:
                pass
        ss = []
        if self.minibatch_metrics is not None:
            ss.append("%.6f" % (self.epoch_metrics[minibatch_class][0]))
        if self.minibatch_n_err is not None:
            ss.append("%.2fpt" % (self.epoch_n_err_pt[minibatch_class]))
        self.fnme = os.path.join(config.root.common.snapshot_dir,
                                 "%s_%s.%d.pickle" %
                                 (self.snapshot_prefix, "_".join(ss),
                                  3 if six.PY3 else 2))
        self.info("Snapshotting to %s" % (self.fnme))
        with open(self.fnme, "wb") as fout:
            pickle.dump(self.workflow, fout)
        fnme_link = os.path.join(config.root.common.snapshot_dir,
                                 "%s_current.%d.pickle" %
                                 (self.snapshot_prefix, 3 if six.PY3 else 2))
        try:
            os.remove(fnme_link)
        except:
            pass
        os.symlink("%s_%s.%d.pickle" % (self.snapshot_prefix,
                                        "_".join(ss), 3 if six.PY3 else 2),
                   fnme_link)

    def _on_export_weights(self, minibatch_class):
        to_rm = []
        if self.fnmeWb is not None:
            to_rm.append("%s.bak" % (self.fnmeWb))
            try:
                os.unlink(to_rm[-1])
            except OSError:
                pass
            try:
                os.rename(self.fnmeWb, to_rm[-1])
            except FileNotFoundError:
                pass
        ss = []
        if self.minibatch_metrics is not None:
            ss.append("%.6f" % (self.epoch_metrics[minibatch_class][0]))
        if self.minibatch_n_err is not None:
            ss.append("%.2fpt" % (self.epoch_n_err_pt[minibatch_class]))
        self.fnmeWb = os.path.join(config.root.common.snapshot_dir,
                                   "%s_%s_Wb.%d.pickle" %
                                   (self.snapshot_prefix, "_".join(ss),
                                    3 if six.PY3 else 2))
        self.info("Exporting weights to %s" % (self.fnmeWb))
        weights = []
        bias = []
        for forward in self.workflow.forward:
            if forward.weights is not None:
                forward.weights.map_read()
                weights.append(forward.weights.v)
            else:
                weights.append(None)
            if forward.bias is not None:
                forward.bias.map_read()
                bias.append(forward.bias.v)
            else:
                bias.append(None)
            if (forward.weights is None or forward.bias is None or
               forward.weights.v is None or forward.bias.v is None):
                continue
            if forward.weights.v.dtype in (numpy.complex64, numpy.complex128):
                self.info("%f %f %f %f" % (
                    min(forward.weights.v.real.min(),
                        forward.weights.v.imag.min()),
                    max(forward.weights.v.real.max(),
                        forward.weights.v.imag.max()),
                    min(forward.bias.v.real.min(),
                        forward.bias.v.imag.min()),
                    max(forward.bias.v.real.max(),
                        forward.bias.v.imag.max())))
            else:
                self.info("%f %f %f %f" % (
                    forward.weights.v.min(), forward.weights.v.max(),
                    forward.bias.v.min(), forward.bias.v.max()))
        with open(self.fnmeWb, "wb") as fout:
            pickle.dump((weights, bias), fout)
        for fnme in to_rm:
            try:
                os.unlink(fnme)
            except OSError:
                pass
        fnme_link = os.path.join(config.root.common.snapshot_dir,
                                 "%s_current_Wb.%d.pickle" %
                                 (self.snapshot_prefix, 3 if six.PY3 else 2))
        try:
            os.remove(fnme_link)
        except:
            pass
        os.symlink("%s_%s_Wb.%d.pickle" % (self.snapshot_prefix,
                                           "_".join(ss), 3 if six.PY3 else 2),
                   fnme_link)

    def _on_export(self, minibatch_class):
        if self.workflow is None:
            return
        if self._do_snapshots:
            self._on_snapshot(minibatch_class)
        if self._do_export_weights:
            self._on_export_weights(minibatch_class)
        self.just_snapshotted << True
        self.snapshot_time = time.time()

    def _on_stop_condition(self, minibatch_class):
        if (((self.epoch_number - self.min_validation_mse_epoch_number >
              self.fail_iterations) and
                self.epoch_number - self.min_validation_n_err_epoch_number >
                self.fail_iterations) or
                self.min_validation_n_err <= 0 or
                self.min_validation_mse <= 0):
            self.complete << True

        # stop if max epoch number reached     333
        if self.max_epochs is not None:
            if self.epoch_number >= self.max_epochs:
                self.complete << True

    def _on_test_validation_processed(self, minibatch_class):
        if self.just_snapshotted:
            self.just_snapshotted << False
        do_snapshot = False
        if (self.minibatch_metrics is not None and
            (self.epoch_min_mse[minibatch_class] < self.min_validation_mse or
             (self.epoch_min_mse[minibatch_class] == self.min_validation_mse
              and self.epoch_min_mse[2] < self.min_train_mse))):
            self.min_validation_mse = self.epoch_min_mse[minibatch_class]
            self.min_validation_mse_epoch_number = self.epoch_number
            self.min_train_mse = self.epoch_min_mse[2]
            do_snapshot = True
        if (self.minibatch_n_err is not None and
            (self.epoch_n_err[minibatch_class] < self.min_validation_n_err or
             (self.epoch_n_err[minibatch_class] == self.min_validation_n_err
              and self.epoch_n_err[2] < self.min_train_n_err))):
            self.min_validation_n_err = self.epoch_n_err[minibatch_class]
            self.min_validation_n_err_epoch_number = self.epoch_number
            self.min_train_n_err = self.epoch_n_err[2]
            do_snapshot = True
        if do_snapshot:
            # Export workflow and weights
            self._on_export(minibatch_class)
        # Stop condition
        self._on_stop_condition(minibatch_class)

    def _print_statistics(self, minibatch_class):
        ss = []
        if self.epoch_metrics[minibatch_class] is not None:
            ss.append("AvgMSE %.6f MaxMSE %.6f "
                      "MinMSE %.3e" % (self.epoch_metrics[minibatch_class][0],
                                       self.epoch_metrics[minibatch_class][1],
                                       self.epoch_metrics[minibatch_class][2]))
        if self.minibatch_n_err is not None:
            ss.append("n_err %d (%.2f%%)" %
                      (self.epoch_n_err[minibatch_class],
                       self.epoch_n_err_pt[minibatch_class]))
        timestamp = time.time()
        self.info("Epoch %d class %s %s in %.2f sec" %
                  (self.epoch_number, CLASS_NAME[minibatch_class],
                   " ".join(ss),
                   timestamp - self.epoch_timestamps[minibatch_class]))
        self.epoch_timestamps[minibatch_class] = timestamp

    def _reset_statistics(self, minibatch_class):
        # Reset statistics per class
        if (self.minibatch_n_err is not None and
                self.minibatch_n_err.v is not None):
            self.minibatch_n_err.map_invalidate()
            self.minibatch_n_err.v[:] = 0
        if (self.minibatch_metrics is not None and
                self.minibatch_metrics.v is not None):
            self.minibatch_metrics.map_invalidate()
            self.minibatch_metrics.v[:] = 0
            self.minibatch_metrics[2] = 1.0e30
        if (len(self.epoch_samples_mse) > minibatch_class and
                self.epoch_samples_mse[minibatch_class] is not None and
                self.epoch_samples_mse[minibatch_class].v is not None):
            self.epoch_samples_mse[minibatch_class].map_invalidate()
            self.tmp_epoch_samples_mse[minibatch_class].map_write()
            self.epoch_samples_mse[minibatch_class].v[:] = (
                self.tmp_epoch_samples_mse[minibatch_class].v[:])
            self.tmp_epoch_samples_mse[minibatch_class].v[:] = 0
        if (self.minibatch_max_err_y_sum is not None and
                self.minibatch_max_err_y_sum.v is not None):
            self.minibatch_max_err_y_sum.map_invalidate()
            self.minibatch_max_err_y_sum.v[:] = 0
        # Reset confusion matrix
        if (self.minibatch_confusion_matrix is not None and
                self.minibatch_confusion_matrix.v is not None):
            self.minibatch_confusion_matrix.map_invalidate()
            self.minibatch_confusion_matrix.v[:] = 0

    def _on_training_finished(self):
        if self.use_dynamic_alpha:
            if (self.minibatch_metrics is not None and
                    self.minibatch_metrics.v is not None):
                this_train_err = self.epoch_metrics[2][0]
            elif (self.minibatch_n_err is not None and
                  self.minibatch_n_err.v is not None):
                this_train_err = self.epoch_n_err[2]
            else:
                this_train_err = self.prev_train_err
            if self.prev_train_err:
                k = this_train_err / self.prev_train_err
            else:
                k = 1.0
            if k < 1.04:
                ak = 1.05
            else:
                ak = 0.7
            self.prev_train_err = this_train_err
            alpha = 0
            for gd in self.workflow.gd:
                if gd is None:
                    continue
                gd.global_alpha = numpy.clip(ak * gd.global_alpha,
                                             0.00001, 0.75)
                if not alpha:
                    alpha = gd.global_alpha
            self.info("new global_alpha: %.6f" % (alpha))
        self._sync_vectors()

    def _sync_vectors(self):
        # Sync vectors
        for vector in self.vectors_to_sync.keys():
            vector.map_read()
        if self.sample_input is not None:
            self.sample_input[:] = self.workflow.forward[0].input[0]
        if self.sample_output is not None:
            self.sample_output[:] = self.workflow.forward[-1].output[0]
        ev = self.ev if self.ev is not None else self.workflow.ev
        if self.sample_target is not None:
            self.sample_target[:] = ev.target[0]
        if (ev.__dict__.get("labels") in
                self.vectors_to_sync.keys()):
            self.sample_label = ev.labels[0]

    def _on_last_minibatch(self, minibatch_class):
        # Copy confusion matrix
        if (self.minibatch_confusion_matrix is not None and
                self.minibatch_confusion_matrix.v is not None):
            self.minibatch_confusion_matrix.map_read()
            self.confusion_matrixes[minibatch_class][:] = (
                self.minibatch_confusion_matrix.v[:])

        if (self.minibatch_metrics is not None and
                self.minibatch_metrics.v is not None):
            self.minibatch_metrics.map_read()
            self.epoch_min_mse[minibatch_class] = (
                min(self.minibatch_metrics[0] /
                    self.class_samples[minibatch_class],
                    self.epoch_min_mse[minibatch_class]))
            # Copy metrics
            self.epoch_metrics[minibatch_class][:] = (
                self.minibatch_metrics.v[:])
            # Compute average mse
            self.epoch_metrics[minibatch_class][0] = (
                self.epoch_metrics[minibatch_class][0] /
                self.class_samples[minibatch_class])

        if (self.minibatch_n_err is not None and
                self.minibatch_n_err.v is not None):
            self.minibatch_n_err.map_read()
            self.epoch_n_err[minibatch_class] = self.minibatch_n_err[0]
            # Compute error in percents
            if self.class_samples[minibatch_class]:
                self.epoch_n_err_pt[minibatch_class] = (
                    100.0 * self.epoch_n_err[minibatch_class] /
                    self.class_samples[minibatch_class])

        # Store maximum of backpropagated gradient
        if (self.minibatch_max_err_y_sum is not None and
                self.minibatch_max_err_y_sum.v is not None):
            self.minibatch_max_err_y_sum.map_read()
            self.max_err_y_sums[minibatch_class] = (
                self.minibatch_max_err_y_sum[0])

        # Test and Validation sets processed
        if ((self.class_samples[VALID] and minibatch_class == VALID) or
                (not self.class_samples[VALID] and minibatch_class >= VALID)):
            self._on_test_validation_processed(minibatch_class)

        # Training set processed
        if self.minibatch_class == TRAIN:
            self._on_training_finished()

        # Print some statistics
        self._print_statistics(minibatch_class)
        self._reset_statistics(minibatch_class)

        if all(self.no_more_minibatches_left):
            self._end_epoch()

    def _end_epoch(self):
        assert all(self.no_more_minibatches_left)
        if not self.is_slave:
            self.epoch_ended << True
            self.epoch_number += 1
        # Reset n_err
        for i in range(len(self.epoch_n_err)):
            self.epoch_n_err[i] = 0

    def _copy_minibatch_mse(self, minibatch_class, minibatch_size,
                            minibatch_offset):
        if self.epoch_samples_mse:
            self.minibatch_mse.map_read()
            offset = minibatch_offset
            for i in range(minibatch_class):
                offset -= self.class_samples[i]
                size = minibatch_size
            self.tmp_epoch_samples_mse[minibatch_class].map_write()
            self.tmp_epoch_samples_mse[minibatch_class][
                offset:offset + size] = self.minibatch_mse[:size]

    def generate_data_for_master(self):
        self._sync_vectors()
        data = {}
        data["minibatch_class"] = self.minibatch_class
        data["minibatch_size"] = self.minibatch_size
        data["minibatch_offset"] = self.minibatch_offset
        for attr in ["minibatch_n_err", "minibatch_metrics",
                     "minibatch_max_err_y_sum", "minibatch_confusion_matrix"]:
            attrval = getattr(self, attr)
            if attrval:
                attrval.map_read()
                data[attr] = attrval.v
        if self.minibatch_mse:
            self.tmp_epoch_samples_mse[self.minibatch_class].map_read()
            data["minibatch_mse"] = self.tmp_epoch_samples_mse[
                self.minibatch_class].v[:self.minibatch_size]
        data["sample_input"] = self.sample_input
        data["sample_output"] = self.sample_output
        data["sample_target"] = self.sample_target
        data["sample_label"] = self.sample_label
        return data

    def generate_data_for_slave(self, slave=None):
        sid = slave.id
        assert self.slave_minibatch_class_.get(sid) == None
        self.slave_minibatch_class_[sid] = self.minibatch_class
        self.minibatches_balance_[self.minibatch_class] += 1
        if all(self.no_more_minibatches_left):
            self.has_data_for_slave = False
        data = {"minibatch_class": self.minibatch_class,
                "minibatch_offset": self.minibatch_offset,
                "minibatch_size": self.minibatch_size}
        return data

    def apply_data_from_master(self, data):
        self.__dict__.update(data)
        self._reset_statistics_(self.minibatch_class)
        # Prevent doing snapshot and set complete after one epoch
        self.complete << False
        self.epoch_ended << False
        self.min_validation_n_err = 0
        self.min_train_n_err = 0
        self.min_validation_mse = 0
        self.min_train_mse = 0

    def apply_data_from_slave(self, data, slave=None):
        if (self.minibatch_n_err is not None and
                self.minibatch_n_err.v is not None):
            self.minibatch_n_err.map_write()
            self.minibatch_n_err.v += data["minibatch_n_err"]
        if self.minibatch_metrics:
            self.minibatch_metrics.map_write()
            self.minibatch_metrics[0] += data["minibatch_metrics"][0]
            self.minibatch_metrics[1] = max(self.minibatch_metrics[1],
                                            data["minibatch_metrics"][1])
            self.minibatch_metrics[2] = min(self.minibatch_metrics[2],
                                            data["minibatch_metrics"][2])
        if self.minibatch_mse:
            self.minibatch_mse.map_write()
            self.minibatch_mse.v = data["minibatch_mse"]
        if self.minibatch_max_err_y_sum:
            self.minibatch_max_err_y_sum.map_write()
            numpy.maximum(self.minibatch_max_err_y_sum.v,
                          data["minibatch_max_err_y_sum"],
                          self.minibatch_max_err_y_sum.v)
        if self.minibatch_confusion_matrix:
            self.minibatch_confusion_matrix.map_write()
            self.minibatch_confusion_matrix.v += data[
                "minibatch_confusion_matrix"]
        if data["sample_input"] is not None:
            for i, d in enumerate(data["sample_input"]):
                self.sample_input[i] = d
        if data["sample_output"] is not None:
            for i, d in enumerate(data["sample_output"]):
                self.sample_output[i] = d
        if data["sample_target"] is not None:
            for i, d in enumerate(data["sample_target"]):
                self.sample_target[i] = d
        if data["sample_label"] is not None:
            for i, d in enumerate(data["sample_label"]):
                self.sample_label[i] = d
        minibatch_class = data["minibatch_class"]
        minibatch_size = data["minibatch_size"]
        minibatch_offset = data["minibatch_offset"]
        assert minibatch_class == self.slave_minibatch_class_[slave.id]
        self._copy_minibatch_mse(minibatch_class, minibatch_size,
                                 minibatch_offset)
        self.epoch_ended << False
        self._finalize_job(slave)
        if (self.no_more_minibatches_left[minibatch_class] and
            self.minibatches_balance_[minibatch_class] == 0):
            self._on_last_minibatch(minibatch_class)
        if (all(self.no_more_minibatches_left) and
            not any(self.minibatches_balance_) and
            not self.complete):
            self.has_data_for_slave = True

    def _finalize_job(self, slave=None):
        minibatch_class = self.slave_minibatch_class_[slave.id]
        if minibatch_class is None:
            # Slave has dropped while waiting for a new job
            return
        self.minibatches_balance_[minibatch_class] -= 1
        if self.minibatches_balance_[minibatch_class] < 0:
            self.warning("Slave %s resulted in negative minibatch balance",
                         slave.id)
            self.minibatches_balance_[minibatch_class] = 0
        self.slave_minibatch_class_[slave.id] = None

    def drop_slave(self, slave=None):
        self._finalize_job(slave)
