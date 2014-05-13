#!/usr/bin/python3.3 -O
"""
Created on Jul 3, 2013

Cifar all2all.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import pickle

from veles.config import root
import veles.formats as formats
from veles.mutable import Bool
import veles.plotting_units as plotting_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.znicz.loader as loader
import veles.znicz.nn_units as nn_units
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.accumulator as accumulator

train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.defaults = {"decision": {"fail_iterations": 100,
                              "snapshot_prefix": "cifar"},
                 "image_saver": {"out_dirs":
                                 [os.path.join(root.common.cache_dir,
                                               "tmp/test"),
                                  os.path.join(root.common.cache_dir,
                                               "tmp/validation"),
                                  os.path.join(root.common.cache_dir,
                                               "tmp/train")]},
                 "loader": {"minibatch_maxsize": 180},
                 "accumulator": {"n_bars": 30},
                 "weights_plotter": {"limit": 25},
                 "cifar": {"learning_rate": 0.1,
                           "weights_decay": 0.00005,
                           "layers": [100, 10],
                           "path_for_load_data": {"train": train_dir,
                                                  "validation":
                                                  validation_dir}}}


class Loader(loader.FullBatchLoader):
    """Loads Cifar dataset.
    """
    def load_data(self):
        """Here we will load data.
        """
        self.original_data = numpy.zeros([60000, 32, 32, 3],
                                         dtype=numpy.float32)
        self.original_labels = numpy.zeros(60000, dtype=numpy.int32)

        # Load Validation
        fin = open(root.cifar.path_for_load_data.validation, "rb")
        u = pickle._Unpickler(fin)
        u.encoding = 'latin1'
        vle = u.load()
        fin.close()
        self.original_data[:10000] = formats.interleave(
            vle["data"].reshape(10000, 3, 32, 32))[:]
        self.original_labels[:10000] = vle["labels"][:]

        # Load Train
        for i in range(1, 6):
            fin = open(os.path.join(root.cifar.path_for_load_data.train,
                                    ("data_batch_%d" % i)), "rb")
            u = pickle._Unpickler(fin)
            u.encoding = 'latin1'
            vle = u.load()
            fin.close()
            self.original_data[i * 10000: (i + 1) * 10000] = (
                formats.interleave(vle["data"].reshape(10000, 3, 32, 32))[:])
            self.original_labels[i * 10000: (i + 1) * 10000] = vle["labels"][:]

        self.class_samples[0] = 0
        self.nextclass_offsets[0] = 0
        self.class_samples[1] = 10000
        self.nextclass_offsets[1] = 10000
        self.class_samples[2] = 50000
        self.nextclass_offsets[2] = 60000

        self.total_samples = self.original_data.shape[0]

        for sample in self.original_data:
            formats.normalize(sample)


class Workflow(nn_units.NNWorkflow):
    """Sample workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.fwds[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                         device=device)
            else:
                aa = all2all.All2AllSoftmax(self, output_shape=[layers[i]],
                                            device=device)
            self.fwds.append(aa)
            if i:
                self.fwds[i].link_from(self.fwds[i - 1])
                self.fwds[i].link_attrs(self.fwds[i - 1],
                                        ("input", "output"))
            else:
                self.fwds[i].link_from(self.loader)
                self.fwds[i].link_attrs(self.loader,
                                        ("input", "minibatch_data"))

        # Add Accumulator units
        self.accumulator = []
        for i in range(0, len(layers)):
            accum = accumulator.RangeAccumulator(self,
                                                 bars=root.accumulator.n_bars)
            self.accumulator.append(accum)
            if i:
                self.accumulator[i].link_from(self.accumulator[i - 1])
            else:
                self.accumulator[i].link_from(self.fwds[-1])

            self.accumulator[i].link_attrs(self.fwds[i],
                                           ("input", "output"))

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(
            self, out_dirs=root.image_saver.out_dirs)
        self.image_saver.link_from(self.accumulator[-1])
        self.image_saver.link_attrs(self.fwds[-1],
                                    "output", "max_idx")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indexes", "minibatch_indexes"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.image_saver)
        self.evaluator.link_attrs(self.fwds[-1], ("y", "output"), "max_idx")
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("labels", "minibatch_labels"),
                                  ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.Decision(
            self, fail_iterations=root.decision.fail_iterations,
            snapshot_prefix=root.decision.snapshot_prefix)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "no_more_minibatches_left",
                                 "class_samples")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"))

        self.image_saver.gate_skip = ~self.decision.just_snapshotted
        self.image_saver.link_attrs(self.decision,
                                    ("this_save_time", "snapshot_time"))
        for i in range(0, len(layers)):
            self.accumulator[i].reset_flag = ~self.decision.epoch_ended

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(None for i in range(0, len(self.fwds)))
        self.gds[-1] = gd.GDSM(self, device=device)
        self.gds[-1].link_from(self.decision)
        self.gds[-1].link_attrs(self.fwds[-1],
                                ("y", "output"),
                                ("h", "input"),
                                "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_y")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.fwds[i],
                                   ("y", "output"),
                                   ("h", "input"),
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader, ("batch_size",
                                                 "minibatch_size"))
            self.gds[i].link_attrs(self.gds[i + 1], ("err_y", "err_h"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.repeater.link_from(self.gds[0])

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
        # Matrix plotter

        self.decision.vectors_to_sync[self.gds[0].weights] = 1
        self.plt_w = nn_plotting_units.Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit)
        self.plt_w.link_attrs(self.gds[0], ("input", "weights"))
        self.plt_w.link_attrs(self.fwds[0], ("get_shape_from", "input"))
        self.plt_w.input_field = "v"
        self.plt_w.link_from(self.decision)
        self.plt_w.gate_block = ~self.decision.epoch_ended

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

        # MultiHistogram plotter
        self.plt_multi_hist = []
        for i in range(0, len(layers)):
            multi_hist = plotting_units.MultiHistogram(
                self, name="Histogram weights %s" % (i + 1))
            self.plt_multi_hist.append(multi_hist)
            self.plt_multi_hist[i].link_from(self.decision)
            self.plt_multi_hist[i].link_attrs(self.fwds[i],
                                              ("input", "weights"))
            self.plt_multi_hist[i].hist_number = self.fwds[
                i].output_shape[0]
            self.plt_multi_hist[i].gate_block = ~self.decision.epoch_ended

    def initialize(self, learning_rate, weights_decay, minibatch_maxsize,
                   device):
        super(Workflow, self).initialize(learning_rate=learning_rate,
                                         weights_decay=weights_decay,
                                         minibatch_maxsize=minibatch_maxsize,
                                         device=device)


def run(load, main):
    load(Workflow, layers=root.cifar.layers)
    main(learning_rate=root.cifar.learning_rate,
         weights_decay=root.cifar.weights_decay,
         minibatch_maxsize=root.loader.minibatch_maxsize)
