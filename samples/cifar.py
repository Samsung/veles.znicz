#!/usr/bin/python3.3 -O
"""
Created on Jul 3, 2013

Cifar all2all.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy
import os
import pickle

from veles.config import root, get
import veles.formats as formats
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.znicz.loader as loader
import veles.znicz.nn_units as nn_units
import veles.znicz.accumulator as accumulator

root.update = {"decision": {"fail_iterations":
                            get(root.decision.fail_iterations, 100),
                            "snapshot_prefix":
                            get(root.decision.snapshot_prefix,
                                       "cifar")},
               "global_alpha": get(root.global_alpha, 0.1),
               "global_lambda": get(root.global_lambda, 0.00005),
               "image_saver": {"out":
                               get(root.path_for_out_data,
                                          os.path.join(root.common.cache_dir,
                                                       "tmp/"))},
               "layers": get(root.layers, [100, 10]),
               "loader": {"minibatch_maxsize":
                          get(root.loader.minibatch_maxsize, 180)},
               "n_bars": get(root.n_bars, 30),
               "path_for_train_data":
               get(root.path_for_train_data,
                          os.path.join(root.common.test_dataset_root,
                                       "cifar/10")),
               "path_for_valid_data":
               get(root.path_for_valid_data,
                          os.path.join(root.common.test_dataset_root,
                                       "cifar/10/test_batch")),
               "weights_plotter": {"limit":
                                   get(root.weights_plotter.limit, 25)}
               }


class Loader(loader.FullBatchLoader):
    """Loads Cifar dataset.
    """
    def load_data(self):
        """Here we will load data.
        """
        n_classes = 10
        self.original_data = numpy.zeros([60000, 32, 32, 3],
                                         dtype=numpy.float32)
        self.original_labels = numpy.zeros(
            60000, dtype=opencl_types.itypes[
                opencl_types.get_itype_from_size(n_classes)])

        # Load Validation
        fin = open(root.path_for_valid_data, "rb")
        u = pickle._Unpickler(fin)
        u.encoding = 'latin1'
        vle = u.load()
        fin.close()
        self.original_data[:10000] = formats.interleave(
            vle["data"].reshape(10000, 3, 32, 32))[:]
        self.original_labels[:10000] = vle["labels"][:]

        # Load Train
        for i in range(1, 6):
            fin = open(os.path.join(root.path_for_train_data,
                                    ("data_batch_%d" % i)), "rb")
            u = pickle._Unpickler(fin)
            u.encoding = 'latin1'
            vle = u.load()
            fin.close()
            self.original_data[i * 10000: (i + 1) * 10000] = (
                formats.interleave(vle["data"].reshape(10000, 3, 32, 32))[:])
            self.original_labels[i * 10000: (i + 1) * 10000] = vle["labels"][:]

        self.class_samples[0] = 0
        self.nextclass_offs[0] = 0
        self.class_samples[1] = 10000
        self.nextclass_offs[1] = 10000
        self.class_samples[2] = 50000
        self.nextclass_offs[2] = 60000

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

        self.rpt.link_from(self.start_point)

        self.loader = Loader(self,
                             minibatch_maxsize=root.loader.minibatch_maxsize)
        self.loader.link_from(self.rpt)

        # Add forward units
        del self.forward[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                         device=device)
            else:
                aa = all2all.All2AllSoftmax(self, output_shape=[layers[i]],
                                            device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add Accumulator units
        self.accumulator = []
        for i in range(0, len(layers)):
            accum = accumulator.RangeAccumulator(self, bars=root.n_bars)
            self.accumulator.append(accum)
            if i:
                self.accumulator[i].link_from(self.accumulator[i - 1])
            else:
                self.accumulator[i].link_from(self.forward[-1])

            self.accumulator[i].link_attrs(self.forward[i],
                                           ("input", "output"))

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self)
        self.image_saver.link_from(self.accumulator[-1])
        self.image_saver.link_attrs(self.forward[-1],
                                    "output", "max_idx")
        self.image_saver.link_attrs(self.loader,
                                    ("input", "minibatch_data"),
                                    ("indexes", "minibatch_indexes"),
                                    ("labels", "minibatch_labels"),
                                    "minibatch_class", "minibatch_size")

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(self, device=device)
        self.ev.link_from(self.image_saver)
        self.ev.link_attrs(self.forward[-1], ("y", "output"), "max_idx")
        self.ev.link_attrs(self.loader,
                           ("batch_size", "minibatch_size"),
                           ("labels", "minibatch_labels"),
                           ("max_samples_per_epoch", "total_samples"))

        # Add decision unit
        self.decision = decision.Decision(
            self, fail_iterations=root.decision.fail_iterations,
            snapshot_prefix=root.decision.snapshot_prefix)
        self.decision.link_from(self.ev)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "minibatch_last",
                                 "class_samples")
        self.decision.link_attrs(
            self.ev,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"))

        self.image_saver.gate_skip = ~self.decision.just_snapshotted
        self.image_saver.link_attrs(self.decision,
                                    ("this_save_time", "snapshot_time"))
        for i in range(0, len(layers)):
            self.accumulator[i].reset_flag = ~self.decision.epoch_ended

        # Add gradient descent units
        del self.gd[:]
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(self, device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].link_attrs(self.forward[-1],
                               ("y", "output"),
                               ("h", "input"),
                               "weights", "bias")
        self.gd[-1].link_attrs(self.ev, "err_y")
        self.gd[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gd[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(self, device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].link_attrs(self.forward[i],
                                  ("y", "output"),
                                  ("h", "input"),
                                  "weights", "bias")
            self.gd[i].link_attrs(self.loader, ("batch_size",
                                                "minibatch_size"))
            self.gd[i].link_attrs(self.gd[i + 1], ("err_y", "err_h"))
            self.gd[i].gate_skip = self.decision.gd_skip
        self.rpt.link_from(self.gd[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Matrix plotter
        # """
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_w = plotting_units.Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit)
        self.plt_w.input = self.gd[0].weights
        self.plt_w.get_shape_from = self.forward[0].input
        self.plt_w.input_field = "v"
        self.plt_w.link_from(self.decision)
        self.plt_w.gate_block = ~self.decision.epoch_ended
        # """
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(0, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
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
            self.plt_hist[i].input = self.accumulator[i].output
            self.plt_hist[i].n_bars = self.accumulator[i].n_bars
            self.plt_hist[i].x = self.accumulator[i].input
            self.plt_hist[i].gate_block = ~self.decision.epoch_ended

        # MultiHistogram plotter
        self.plt_multi_hist = []
        for i in range(0, len(layers)):
            multi_hist = plotting_units.MultiHistogram(
                self, name="Histogram weights %s" % (i + 1))
            self.plt_multi_hist.append(multi_hist)
            self.plt_multi_hist[i].link_from(self.decision)
            self.plt_multi_hist[i].hist_number = self.forward[
                i].output_shape[0]
            self.plt_multi_hist[i].input = self.forward[i].weights
            self.plt_multi_hist[i].gate_block = ~self.decision.epoch_ended

    def initialize(self, global_alpha, global_lambda, minibatch_maxsize,
                   device):
        self.loader.minibatch_maxsize = minibatch_maxsize
        self.ev.device = device
        for g in self.gd:
            g.device = device
            g.global_alpha = global_alpha
            g.global_lambda = global_lambda
        for forward in self.forward:
            forward.device = device
        return super(Workflow, self).initialize()


def run(load, main):
    load(Workflow, layers=root.layers)
    main(global_alpha=root.global_alpha, global_lambda=root.global_lambda,
         minibatch_maxsize=root.loader.minibatch_maxsize)
