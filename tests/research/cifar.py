#!/usr/bin/python3.3 -O
"""
Created on Jul 3, 2013

Cifar all2all.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import pickle
import six
from zope.interface import implementer

from veles.config import root
import veles.formats as formats
from veles.mutable import Bool
import veles.plotting_units as plotting_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.conv as conv
import veles.znicz.evaluator as evaluator
import veles.znicz.image_saver as image_saver
import veles.znicz.learning_rate_adjust as learning_rate_adjust
import veles.znicz.loader as loader
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.nn_units import NNSnapshotter
from veles.znicz.standard_workflow import StandardWorkflow


train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.defaults = {
    "accumulator": {"bars": 30},
    "decision": {"fail_iterations": 100, "do_export_weights": True},
    "snapshotter": {"prefix": "cifar"},
    "softmax": {"error_function_avr": False},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 180, "norm": "-1, 1",
               "shuffle_limit": 1},
    "weights_plotter": {"limit": 64},
    "cifar": {"layers": [{"type": "all2all_tanh",
                          "learning_rate": 0.0005,
                          "weights_decay": 0.00005,
                          "output_shape": 100},
                         {"type": "softmax", "output_shape": 10,
                          "learning_rate": 0.0005,
                          "weights_decay": 0.00005}],
              "data_paths": {"train": train_dir,
                             "validation": validation_dir}}}


@implementer(loader.IFullBatchLoader)
class Loader(loader.FullBatchLoader):
    """Loads Cifar dataset.
    """
    def __init__(self, workflow, **kwargs):
        super(Loader, self).__init__(workflow, **kwargs)
        self.shuffle_limit = kwargs.get("shuffle_limit", 2000000000)

    def shuffle(self):
        if self.shuffle_limit <= 0:
            return
        self.shuffle_limit -= 1
        super(Loader, self).shuffle()

    def load_data(self):
        """Here we will load data.
        """
        self.original_data = numpy.zeros([60000, 32, 32, 3],
                                         dtype=numpy.float32)
        self.original_labels = numpy.zeros(60000, dtype=numpy.int32)

        # Load Validation
        with open(root.cifar.data_paths.validation, "rb") as fin:
            if six.PY3:
                vle = pickle.load(fin, encoding='latin1')
            else:
                vle = pickle.load(fin)
        fin.close()
        self.original_data[:10000] = formats.interleave(
            vle["data"].reshape(10000, 3, 32, 32))[:]
        self.original_labels[:10000] = vle["labels"][:]

        # Load Train
        for i in range(1, 6):
            with open(os.path.join(root.cifar.data_paths.train,
                                   ("data_batch_%d" % i)), "rb") as fin:
                if six.PY3:
                    vle = pickle.load(fin, encoding='latin1')
                else:
                    vle = pickle.load(fin)
            self.original_data[i * 10000: (i + 1) * 10000] = (
                formats.interleave(vle["data"].reshape(10000, 3, 32, 32))[:])
            self.original_labels[i * 10000: (i + 1) * 10000] = vle["labels"][:]

        self.class_lengths[0] = 0
        self.class_offsets[0] = 0
        self.class_lengths[1] = 10000
        self.class_offsets[1] = 10000
        self.class_lengths[2] = 50000
        self.class_offsets[2] = 60000

        self.total_samples = self.original_data.shape[0]

        if root.loader.norm == "mean":
            mean = numpy.mean(self.original_data[10000:], axis=0)
            self.original_data -= mean
            self.info("Validation range: %.6f %.6f %.6f",
                      self.original_data[:10000].min(),
                      self.original_data[:10000].max(),
                      numpy.average(self.original_data[:10000]))
            self.info("Train range: %.6f %.6f %.6f",
                      self.original_data[10000:].min(),
                      self.original_data[10000:].max(),
                      numpy.average(self.original_data[10000:]))
        elif root.loader.norm == "-1, 1":
            for sample in self.original_data:
                formats.normalize(sample)
        elif root.loader.norm == "-128, 128":
            for sample in self.original_data:
                formats.normalize(sample)
                sample *= 128
        else:
            raise ValueError("Unsupported normalization type "
                             + str(root.loader.norm))


class Workflow(StandardWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, shuffle_limit=root.loader.shuffle_limit)
        self.loader.link_from(self.repeater)

        # Add fwds units
        self.parse_forwards_from_config()

        # Add Accumulator units
        """
        self.accumulator = []
        for i in range(0, len(layers)):
            accum = accumulator.RangeAccumulator(self,
                                                 bars=root.accumulator.bars)
            self.accumulator.append(accum)
            if i:
                self.accumulator[i].link_from(self.accumulator[i - 1])
            else:
                self.accumulator[i].link_from(self.fwds[-1])

            self.accumulator[i].link_attrs(self.fwds[i],
                                           ("input", "output"))
        """
        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(
            self, out_dirs=root.image_saver.out_dirs)
        # self.image_saver.link_from(self.accumulator[-1])
        self.image_saver.link_from(self.fwds[-1])
        self.image_saver.link_attrs(self.fwds[-1],
                                    "output", "max_idx")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indexes", "minibatch_indices"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.image_saver)
        self.evaluator.link_attrs(self.fwds[-1], "output", "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("labels", "minibatch_labels"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.DecisionGD(
            self, fail_iterations=root.decision.fail_iterations,
            do_export_weights=root.decision.do_export_weights)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "last_minibatch",
                                 "minibatch_size",
                                 "class_lengths",
                                 "epoch_ended",
                                 "epoch_number")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"),
            ("minibatch_max_err_y_sum", "max_err_output_sum"))
        self.decision.fwds = self.fwds
        self.decision.gds = self.gds
        self.decision.evaluator = self.evaluator

        self.snapshotter = NNSnapshotter(self, prefix=root.snapshotter.prefix,
                                         directory=root.common.snapshot_dir)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.decision.epoch_ended | ~self.decision.improved)

        self.image_saver.gate_skip = ~self.decision.improved
        self.image_saver.link_attrs(self.snapshotter,
                                    ("this_save_time", "time"))
        # for i in range(0, len(layers)):
        #    self.accumulator[i].reset_flag = ~self.decision.epoch_ended

        # Add gradient descent units
        self.create_gradient_descent_units()

        # Add learning_rate_adjust unit
        self.learning_rate_adjust = learning_rate_adjust.LearningRateAdjust(
            self,
            lr_function=learning_rate_adjust.arbitrary_step_function_policy(
                [(0.001, 60000), (0.0001, 65000), (0.00001, 70000)]))
        self.learning_rate_adjust.link_from(self.gds[0])
        self.learning_rate_adjust.add_gd_units(self.gds)

        self.repeater.link_from(self.learning_rate_adjust)

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_n_err_pt"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True

        """
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(0, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].link_attrs(self.decision,
                                       ("input", "confusion_matrixes"))
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.plt[-1])
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended

        # Histogram plotter
        self.plt_hist = []
        for i in range(0, len(layers)):
            hist = plotting_units.Histogram(self, name="Histogram output %s" %
                                            (i + 1))
            self.plt_hist.append(hist)
            self.plt_hist[i].link_from(self.decision)
            self.plt_hist[i].link_attrs(self.accumulator[i],
                                        ("input", "output"),
                                        ("x", "input"), "n_bars")
            self.plt_hist[i].gate_block = ~self.decision.epoch_ended
        """

        # Weights plotter
        self.plt_mx = []
        prev_channels = 3
        for i in range(0, len(layers)):
            if (not isinstance(self.fwds[i], conv.Conv) and
                    not isinstance(self.fwds[i], all2all.All2All)):
                continue
            self.decision.vectors_to_sync[self.fwds[i].weights] = 1
            plt_mx = nn_plotting_units.Weights2D(
                self, name="%s %s" % (i + 1, layers[i]["type"]),
                limit=root.weights_plotter.limit)
            self.plt_mx.append(plt_mx)
            self.plt_mx[-1].link_attrs(self.fwds[i], ("input", "weights"))
            self.plt_mx[-1].input_field = "mem"
            if isinstance(self.fwds[i], conv.Conv):
                self.plt_mx[-1].get_shape_from = (
                    [self.fwds[i].kx, self.fwds[i].ky, prev_channels])
                prev_channels = self.fwds[i].n_kernels
            if (layers[i].get("output_shape") is not None and
                    layers[i]["type"] != "softmax"):
                self.plt_mx[-1].link_attrs(self.fwds[i],
                                           ("get_shape_from", "input"))
            # elif isinstance(self.fwds[i], all2all.All2All):
            #    self.plt_mx[-1].get_shape_from = self.fwds[i].input
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended

        # MultiHistogram plotter
        self.plt_multi_hist = []
        for i in range(0, len(layers)):
            multi_hist = plotting_units.MultiHistogram(
                self, name="Histogram %s %s" % (i + 1, layers[i]["type"]))
            self.plt_multi_hist.append(multi_hist)
            if layers[i].get("n_kernels") is not None:
                self.plt_multi_hist[i].link_from(self.decision)
                self.plt_multi_hist[i].hist_number = layers[i]["n_kernels"]
                self.plt_multi_hist[i].link_attrs(self.fwds[i],
                                                  ("input", "weights"))
                end_epoch = ~self.decision.epoch_ended
                self.plt_multi_hist[i].gate_block = end_epoch
            if layers[i].get("output_shape") is not None:
                self.plt_multi_hist[i].link_from(self.decision)
                self.plt_multi_hist[i].hist_number = layers[i]["output_shape"]
                self.plt_multi_hist[i].link_attrs(self.fwds[i],
                                                  ("input", "weights"))
                self.plt_multi_hist[i].gate_block = ~self.decision.epoch_ended

        # Table plotter
        self.plt_tab = plotting_units.TableMaxMin(self, name="Max, Min")
        del self.plt_tab.y[:]
        del self.plt_tab.col_labels[:]
        for i in range(0, len(layers)):
            if (not isinstance(self.fwds[i], conv.Conv) and
                    not isinstance(self.fwds[i], all2all.All2All)):
                continue
            obj = self.fwds[i].weights
            name = "weights %s %s" % (i + 1, layers[i]["type"])
            self.plt_tab.y.append(obj)
            self.plt_tab.col_labels.append(name)
            obj = self.gds[i].gradient_weights
            name = "gd %s %s" % (i + 1, layers[i]["type"])
            self.plt_tab.y.append(obj)
            self.plt_tab.col_labels.append(name)
            obj = self.fwds[i].output
            name = "Y %s %s" % (i + 1, layers[i]["type"])
            self.plt_tab.y.append(obj)
            self.plt_tab.col_labels.append(name)
        self.plt_tab.link_from(self.decision)
        self.plt_tab.gate_block = ~self.decision.epoch_ended

    def initialize(self, minibatch_size, device):
        super(Workflow, self).initialize(minibatch_size=minibatch_size,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.cifar.layers)
    main(minibatch_size=root.loader.minibatch_size)
