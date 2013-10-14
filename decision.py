"""
Created on Aug 15, 2013

Decision unit.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import units
import formats
import config
import time
import os
import pickle


class Decision(units.Unit):
    """Decides on the learning behavior.

    Attributes:
        complete: completed.
        minibatch_class: current minibatch class.
        minibatch_last: if current minibatch is last in it's class.
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
        epoch_metrics: metrics for an epoch (same as minibatch_metrics).
        workflow: reference to workflow to snapshot.
        snapshot_prefix: prefix for the snapshots.
        fnme: filename of the last snapshot.
        fnmeWb: filename of the last weights + bias snapshot.
        just_snapshotted: 1 after snapshot.
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
                      if the later is in vectors_to_sync.
        sample_output: will be a copy of first element from forward[-1].output
                       if the later is in vectors_to_sync.
        sample_target: will be a copy of first element from ev.target
                       if the later is in vectors_to_sync.
        sample_label: will be a copy of first element label from ev.labels
                       if the later is in vectors_to_sync.
        use_dynamic_alpha: will adjust alpha according to previous train error.
    """
    def __init__(self, fail_iterations=100, snapshot_prefix="",
                 store_samples_mse=False, use_dynamic_alpha=False):
        super(Decision, self).__init__()
        self.minibatch_class = None  # [0]
        self.minibatch_last = None  # [0]
        self.class_samples = None  # [0, 0, 0]
        self.fail_iterations = [fail_iterations]
        self.complete = [0]
        self.gd_skip = [0]
        self.epoch_number = [0]
        self.epoch_ended = [0]
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
        self.workflow = None
        self.snapshot_prefix = snapshot_prefix
        self.fnme = None
        self.fnmeWb = None
        self.t1 = None
        self.epoch_metrics = [None, None, None]
        self.just_snapshotted = [0]
        self.snapshot_time = [0]
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
        self.minibatch_offs = None
        self.minibatch_size = None
        self.confusion_matrixes = [None, None, None]
        self.minibatch_confusion_matrix = None  # formats.Vector()
        self.max_err_y_sums = [0, 0, 0]
        self.minibatch_max_err_y_sum = None  # formats.Vector()
        self.vectors_to_sync = {}
        self.sample_input = None
        self.sample_output = None
        self.sample_target = None
        self.sample_label = None
        self.use_dynamic_alpha = use_dynamic_alpha
        self.prev_train_err = 1.0e30

    def init_unpickled(self):
        super(Decision, self).init_unpickled()
        self.epoch_min_mse = [1.0e30, 1.0e30, 1.0e30]
        self.epoch_n_err = [1.0e30, 1.0e30, 1.0e30]
        self.epoch_n_err_pt = [100.0, 100.0, 100.0]

    def initialize(self):
        # Allocate arrays for confusion matrixes.
        if (self.minibatch_confusion_matrix != None and
            self.minibatch_confusion_matrix.v != None):
            for i in range(0, len(self.confusion_matrixes)):
                self.confusion_matrixes[i] = (
                    numpy.zeros_like(self.minibatch_confusion_matrix.v))

        # Allocate arrays for epoch metrics.
        if (self.minibatch_metrics != None and
            self.minibatch_metrics.v != None):
            for i in range(0, len(self.epoch_metrics)):
                self.epoch_metrics[i] = (
                    numpy.zeros_like(self.minibatch_metrics.v))

        # Allocate vectors for storing samples mse.
        for i in range(0, len(self.epoch_samples_mse)):
            if self.class_samples[i] <= 0:
                continue
            if (self.tmp_epoch_samples_mse[i].v == None or
                self.tmp_epoch_samples_mse[i].v.size != self.class_samples[i]):
                self.tmp_epoch_samples_mse[i].v = (
                    numpy.zeros(self.class_samples[i],
                    dtype=config.dtypes[config.dtype]))
                self.epoch_samples_mse[i].v = (
                    numpy.zeros(self.class_samples[i],
                    dtype=config.dtypes[config.dtype]))

        # Initialize sample_input, sample_output, sample_target if neccessary
        if self.workflow.forward[0].input in self.vectors_to_sync:
            self.sample_input = numpy.zeros_like(
                self.workflow.forward[0].input.v[0])
        if self.workflow.forward[-1].output in self.vectors_to_sync:
            self.sample_output = numpy.zeros_like(
                self.workflow.forward[-1].output.v[0])
        if (self.workflow.ev.__dict__.get("target") != None
            and self.workflow.ev.target in self.vectors_to_sync
            and self.workflow.ev.target.v != None):
            self.sample_target = numpy.zeros_like(
                self.workflow.ev.target.v[0])

    def on_snapshot(self, minibatch_class):
        if self.workflow == None:
            return
        to_rm = []
        if self.fnme != None:
            to_rm.append("%s.bak" % (self.fnme))
            try:
                os.unlink(to_rm[-1])
            except OSError:
                pass
            try:
                os.rename(self.fnme, to_rm[-1])
            except OSError:
                pass
        if self.fnmeWb != None:
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
        if self.minibatch_metrics != None:
            ss.append("%.6f" % (self.epoch_metrics[minibatch_class][0]))
        if self.minibatch_n_err != None:
            ss.append("%.2fpt" % (self.epoch_n_err_pt[minibatch_class]))
        self.fnme = ("%s/%s_%s.pickle" %
            (config.snapshot_dir, self.snapshot_prefix,
             "_".join(ss)))
        self.log().info("Snapshotting to %s" % (self.fnme))
        fout = open(self.fnme, "wb")
        pickle.dump(self.workflow, fout)
        fout.close()
        self.fnmeWb = ("%s/%s_%s_Wb.pickle" %
            (config.snapshot_dir, self.snapshot_prefix,
             "_".join(ss)))
        self.log().info("Exporting weights to %s" % (self.fnmeWb))
        fout = open(self.fnmeWb, "wb")
        weights = []
        bias = []
        for forward in self.workflow.forward:
            forward.weights.sync()
            forward.bias.sync()
            weights.append(forward.weights.v)
            bias.append(forward.bias.v)
            if forward.weights.v.dtype in (numpy.complex64, numpy.complex128):
                self.log().info("%f %f %f %f" % (
                    min(forward.weights.v.real.min(),
                        forward.weights.v.imag.min()),
                    max(forward.weights.v.real.max(),
                        forward.weights.v.imag.max()),
                    min(forward.bias.v.real.min(),
                        forward.bias.v.imag.min()),
                    max(forward.bias.v.real.max(),
                        forward.bias.v.imag.max())))
            else:
                self.log().info("%f %f %f %f" % (
                    forward.weights.v.min(), forward.weights.v.max(),
                    forward.bias.v.min(), forward.bias.v.max()))
        pickle.dump((weights, bias), fout)
        fout.close()
        for fnme in to_rm:
            try:
                os.unlink(fnme)
            except OSError:
                pass
        self.just_snapshotted[0] = 1
        self.snapshot_time[0] = time.time()

    def on_stop_condition(self, minibatch_class):
        if (self.epoch_number[0] - self.min_validation_mse_epoch_number >
            self.fail_iterations[0] and
            self.epoch_number[0] - self.min_validation_n_err_epoch_number >
            self.fail_iterations[0]) or self.min_validation_n_err == 0:
            self.complete[0] = 1

    def on_test_validation_processed(self, minibatch_class):
        if self.just_snapshotted[0]:
            self.just_snapshotted[0] = 0
        do_snapshot = False
        if (self.minibatch_metrics != None and
            (self.epoch_min_mse[minibatch_class] < self.min_validation_mse or
             (self.epoch_min_mse[minibatch_class] == self.min_validation_mse
              and self.epoch_min_mse[2] < self.min_train_mse))):
            self.min_validation_mse = self.epoch_min_mse[minibatch_class]
            self.min_validation_mse_epoch_number = self.epoch_number[0]
            self.min_train_mse = self.epoch_min_mse[2]
            do_snapshot = True
        if (self.minibatch_n_err != None and
            (self.epoch_n_err[minibatch_class] < self.min_validation_n_err or
             (self.epoch_n_err[minibatch_class] == self.min_validation_n_err
              and self.epoch_n_err[2] == self.min_train_n_err))):
            self.min_validation_n_err = self.epoch_n_err[minibatch_class]
            self.min_validation_n_err_epoch_number = self.epoch_number[0]
            self.min_train_n_err = self.epoch_n_err[2]
            do_snapshot = True
        if do_snapshot:
            # Do the snapshot
            self.on_snapshot(minibatch_class)
        # Stop condition
        self.on_stop_condition(minibatch_class)

    def on_print_statistics(self, minibatch_class, dt):
        ss = []
        if self.epoch_metrics[minibatch_class] != None:
            ss.append("AvgMSE %.6f MaxMSE %.6f "
                      "MinMSE %.3e" % (
                      self.epoch_metrics[minibatch_class][0],
                      self.epoch_metrics[minibatch_class][1],
                      self.epoch_metrics[minibatch_class][2]))
        if self.minibatch_n_err != None:
            ss.append("n_err %d (%.2f%%)" % (
                      self.epoch_n_err[minibatch_class],
                      self.epoch_n_err_pt[minibatch_class]))
        self.log().info("Epoch %d Class %d %s in %.2f sec" % (
                        self.epoch_number[0], minibatch_class,
                        " ".join(ss), dt))

    def on_reset_statistics(self, minibatch_class):
        # Reset statistics per class
        if self.minibatch_n_err != None and self.minibatch_n_err.v != None:
            self.minibatch_n_err.v[:] = 0
            self.minibatch_n_err.update()
        if (self.minibatch_metrics != None and
            self.minibatch_metrics.v != None):
            self.minibatch_metrics.v[:] = 0
            self.minibatch_metrics.v[2] = 1.0e30
            self.minibatch_metrics.update()
        if (len(self.epoch_samples_mse) >= minibatch_class and
            self.epoch_samples_mse[minibatch_class] != None and
            self.epoch_samples_mse[minibatch_class].v != None):
            self.epoch_samples_mse[minibatch_class].v[:] = (
                self.tmp_epoch_samples_mse[minibatch_class].v[:])
            self.tmp_epoch_samples_mse[minibatch_class].v[:] = 0
        if (self.minibatch_max_err_y_sum != None and
            self.minibatch_max_err_y_sum.v != None):
            self.minibatch_max_err_y_sum.v[:] = 0
            self.minibatch_max_err_y_sum.update()
        # Reset confusion matrix
        if (self.minibatch_confusion_matrix != None and
            self.minibatch_confusion_matrix.v != None):
            self.minibatch_confusion_matrix.v[:] = 0
            self.minibatch_confusion_matrix.update()

    def on_training_processed(self, minibatch_class):
        if self.use_dynamic_alpha:
            if (self.minibatch_metrics != None and
                self.minibatch_metrics.v != None):
                this_train_err = self.epoch_metrics[2][0]
            elif (self.minibatch_n_err != None and
                  self.minibatch_n_err.v != None):
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
                if gd == None:
                    continue
                gd.global_alpha = numpy.clip(ak * gd.global_alpha,
                                             0.00001, 0.75)
                if not alpha:
                    alpha = gd.global_alpha
            self.log().info("new global_alpha: %.6f" % (alpha))
        self.epoch_ended[0] = 1
        self.epoch_number[0] += 1
        # Reset n_err
        for i in range(0, len(self.epoch_n_err)):
            self.epoch_n_err[i] = 0
        # Sync vectors
        for vector in self.vectors_to_sync.keys():
            vector.sync()
        if self.sample_input != None:
            self.sample_input[:] = self.workflow.forward[0].input.v[0]
        if self.sample_output != None:
            self.sample_output[:] = self.workflow.forward[-1].output.v[0]
        if self.sample_target != None:
            self.sample_target[:] = self.workflow.ev.target.v[0]
        if (self.workflow.ev.__dict__.get("labels") in
            self.vectors_to_sync.keys()):
            self.sample_label = self.workflow.ev.labels.v[0]

    def on_last_minibatch(self, minibatch_class):
        # Copy confusion matrix
        if (self.minibatch_confusion_matrix != None and
            self.minibatch_confusion_matrix.v != None):
            self.minibatch_confusion_matrix.sync()
            self.confusion_matrixes[minibatch_class][:] = (
                self.minibatch_confusion_matrix.v[:])

        if (self.minibatch_metrics != None and
            self.minibatch_metrics.v != None):
            self.minibatch_metrics.sync()
            self.epoch_min_mse[minibatch_class] = (
                min(self.minibatch_metrics.v[0] /
                    self.class_samples[minibatch_class],
                self.epoch_min_mse[minibatch_class]))
            # Copy metrics
            self.epoch_metrics[minibatch_class][:] = (
                self.minibatch_metrics.v[:])
            # Compute average mse
            self.epoch_metrics[minibatch_class][0] = (
                self.epoch_metrics[minibatch_class][0] /
                self.class_samples[minibatch_class])

        if (self.minibatch_n_err != None and
            self.minibatch_n_err.v != None):
            self.minibatch_n_err.sync()
            self.epoch_n_err[minibatch_class] = self.minibatch_n_err.v[0]
            # Compute error in percents
            if self.class_samples[minibatch_class]:
                self.epoch_n_err_pt[minibatch_class] = (100.0 *
                    self.epoch_n_err[minibatch_class] /
                    self.class_samples[minibatch_class])

        # Store maximum of backpropagated gradient
        if (self.minibatch_max_err_y_sum != None and
            self.minibatch_max_err_y_sum.v != None):
            self.minibatch_max_err_y_sum.sync()
            self.max_err_y_sums[minibatch_class] = (
                self.minibatch_max_err_y_sum.v[0])

        # Test and Validation sets processed
        if ((self.class_samples[1] and minibatch_class == 1) or
            (not self.class_samples[1] and minibatch_class >= 1)):
            self.on_test_validation_processed(minibatch_class)

        # Print some statistics
        t2 = time.time()
        self.on_print_statistics(minibatch_class, t2 - self.t1)
        self.t1 = t2

        # Training set processed
        if self.minibatch_class[0] == 2:
            self.on_training_processed(minibatch_class)

        self.on_reset_statistics(minibatch_class)

    def run(self):
        if self.t1 == None:
            self.t1 = time.time()
        self.complete[0] = 0
        self.epoch_ended[0] = 0

        minibatch_class = self.minibatch_class[0]

        # Copy minibatch mse
        if self.minibatch_mse != None and len(self.epoch_samples_mse):
            self.minibatch_mse.sync()
            offs = self.minibatch_offs[0]
            for i in range(0, minibatch_class):
                offs -= self.class_samples[i]
                size = self.minibatch_size[0]
            self.tmp_epoch_samples_mse[minibatch_class].v[
                offs:offs + size] = self.minibatch_mse.v[:size]

        # Check skip gradient descent or not
        if self.minibatch_class[0] < 2:
            self.gd_skip[0] = 1
        else:
            self.gd_skip[0] = 0

        if self.minibatch_last[0]:
            self.on_last_minibatch(minibatch_class)
