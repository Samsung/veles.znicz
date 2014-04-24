#!/usr/bin/python3.3 -O
"""
Created on Mar 31, 2014

Cifar convolutional.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import pickle

from veles.config import root
import veles.formats as formats
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.conv as conv
import veles.znicz.pooling as pooling
import veles.znicz.gd_conv as gd_conv
import veles.znicz.gd_pooling as gd_pooling
import veles.error as error
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.znicz.loader as loader

train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.defaults = {"decision": {"fail_iterations": 1000,
                              "snapshot_prefix": "cifar_caffe",
                              "do_export_weights": True},
                 "image_saver": {"out_dirs":
                                 [os.path.join(root.common.cache_dir,
                                               "tmp/test"),
                                  os.path.join(root.common.cache_dir,
                                               "tmp/validation"),
                                  os.path.join(root.common.cache_dir,
                                               "tmp/train")]},
                 "loader": {"minibatch_maxsize": 100},
                 "weights_plotter": {"limit": 64},
                 "cifar_caffe": {"global_alpha": 0.001,
                                 "global_lambda": 0.004,
                                 "layers":
                                 [{"type": "conv_relu", "n_kernels": 32,
                                   "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                                  {"type": "max_pooling",
                                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                                  {"type": "conv_relu", "n_kernels": 32,
                                   "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                                  {"type": "avg_pooling",
                                   "kx": 3, "ky": 3, "sliding": (2, 2)},
                                  {"type": "conv_relu", "n_kernels": 64,
                                   "kx": 5, "ky": 5, "padding": (2, 2, 2, 2)},
                                  {"type": "avg_pooling",
                                   "kx": 3, "ky": 3, "sliding": (2, 2)}, 10],
                                 "path_for_load_data": {"train": train_dir,
                                                        "validation":
                                                        validation_dir}}}


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
        fin = open(root.cifar_caffe.path_for_load_data.validation, "rb")
        u = pickle._Unpickler(fin)
        u.encoding = 'latin1'
        vle = u.load()
        fin.close()
        self.original_data[:10000] = formats.interleave(
            vle["data"].reshape(10000, 3, 32, 32))[:]
        self.original_labels[:10000] = vle["labels"][:]

        # Load Train
        for i in range(1, 6):
            fin = open(os.path.join(root.cifar_caffe.path_for_load_data.train,
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
    """Cifar workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.rpt.link_from(self.start_point)

        self.loader = Loader(self)
        self.loader.link_from(self.rpt)

        # Add forward units
        del self.forward[:]
        for i in range(0, len(layers)):
            layer = layers[i]
            if type(layer) == int:
                if i == len(layers) - 1:
                    aa = all2all.All2AllSoftmax(self, output_shape=[layer],
                                                device=device)
                else:
                    aa = all2all.All2AllTanh(self, output_shape=[layer],
                                             device=device)
            elif type(layer) == dict:
                if layer["type"] == "conv":
                    aa = conv.ConvTanh(
                        self, n_kernels=layer["n_kernels"],
                        kx=layer["kx"], ky=layer["ky"],
                        sliding=layer.get("sliding", (1, 1, 1, 1)),
                        padding=layer.get("padding", (0, 0, 0, 0)),
                        device=device)
                elif layer["type"] == "conv_relu":
                    aa = conv.ConvRELU(
                        self, n_kernels=layer["n_kernels"],
                        kx=layer["kx"], ky=layer["ky"],
                        sliding=layer.get("sliding", (1, 1, 1, 1)),
                        padding=layer.get("padding", (0, 0, 0, 0)),
                        device=device,
                        weights_filling="normal")
                elif layer["type"] == "max_pooling":
                    aa = pooling.MaxPooling(
                        self, kx=layer["kx"], ky=layer["ky"],
                        sliding=layer.get("sliding",
                                          (layer["kx"], layer["ky"])),
                        device=device)
                elif layer["type"] == "avg_pooling":
                    aa = pooling.AvgPooling(
                        self, kx=layer["kx"], ky=layer["ky"],
                        sliding=layer.get("sliding",
                                          (layer["kx"], layer["ky"])),
                        device=device)
                else:
                    raise error.ErrBadFormat(
                        "Unsupported layer type %s" % (layer["type"]))
            else:
                raise error.ErrBadFormat(
                    "layers element type should be int "
                    "for all-to-all or dictionary for "
                    "convolutional or pooling")
            self.forward.append(aa)
            if i:
                self.forward[-1].link_from(self.forward[-2])
                self.forward[-1].link_attrs(self.forward[-2],
                                            ("input", "output"))
            else:
                self.forward[-1].link_from(self.loader)
                self.forward[-1].link_attrs(self.loader,
                                            ("input", "minibatch_data"))

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(
            self, out_dirs=root.image_saver.out_dirs)
        self.image_saver.link_from(self.forward[-1])
        self.image_saver.link_attrs(self.forward[-1], "output", "max_idx")
        self.image_saver.link_attrs(
            self.loader,
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
            snapshot_prefix=root.decision.snapshot_prefix,
            do_export_weights=root.decision.do_export_weights)
        self.decision.link_from(self.ev)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class",
                                 "no_more_minibatches_left",
                                 "class_samples")
        self.decision.link_attrs(
            self.ev,
            ("minibatch_n_err", "n_err"),
            ("minibatch_confusion_matrix", "confusion_matrix"))

        self.image_saver.gate_skip = ~self.decision.just_snapshotted
        self.image_saver.link_attrs(self.decision,
                                    ("this_save_time", "snapshot_time"))

        # Add gradient descent units
        del self.gd[:]
        self.gd.extend(list(None for i in range(0, len(self.forward))))
        self.gd[-1] = gd.GDSM(self, device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].link_attrs(self.ev, "err_y")
        self.gd[-1].link_attrs(self.forward[-1],
                               ("y", "output"),
                               ("h", "input"),
                               "weights", "bias")
        self.gd[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gd[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.forward) - 2, -1, -1):
            if isinstance(self.forward[i], conv.ConvTanh):
                obj = gd_conv.GDTanh(
                    self, n_kernels=self.forward[i].n_kernels,
                    kx=self.forward[i].kx, ky=self.forward[i].ky,
                    sliding=self.forward[i].sliding,
                    padding=self.forward[i].padding,
                    device=device)
            elif isinstance(self.forward[i], conv.ConvRELU):
                obj = gd_conv.GDRELU(
                    self, n_kernels=self.forward[i].n_kernels,
                    kx=self.forward[i].kx, ky=self.forward[i].ky,
                    sliding=self.forward[i].sliding,
                    padding=self.forward[i].padding,
                    device=device)
            elif isinstance(self.forward[i], pooling.MaxPooling):
                obj = gd_pooling.GDMaxPooling(
                    self, kx=self.forward[i].kx, ky=self.forward[i].ky,
                    sliding=self.forward[i].sliding,
                    device=device)
                obj.link_attrs(self.forward[i], ("h_offs", "input_offs"))
            elif isinstance(self.forward[i], pooling.AvgPooling):
                obj = gd_pooling.GDAvgPooling(
                    self, kx=self.forward[i].kx, ky=self.forward[i].ky,
                    sliding=self.forward[i].sliding,
                    device=device)
            elif isinstance(self.forward[i], all2all.All2AllTanh):
                obj = gd.GDTanh(self, device=device)
            else:
                raise ValueError("Unsupported forward unit type "
                                 " encountered: %s" %
                                 self.forward[i].__class__.__name__)
            self.gd[i] = obj
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].link_attrs(self.gd[i + 1], ("err_y", "err_h"))
            self.gd[i].link_attrs(self.forward[i],
                                  ("y", "output"),
                                  ("h", "input"),
                                  "weights", "bias")
            self.gd[i].link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"))
            self.gd[i].gate_skip = self.decision.gd_skip

        self.rpt.link_from(self.gd[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision
                                   if len(self.plt) == 1 else self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended
                                       if len(self.plt) == 1 else Bool(False))
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Confusion matrix plotter
        """
        self.plt_mx = []
        for i in range(1, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.plt[-1])
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended
        """
        # Weights plotter
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_mx = plotting_units.Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit)
        self.plt_mx.input = self.gd[0].weights
        self.plt_mx.input_field = "v"
        self.plt_mx.get_shape_from = (
            [self.forward[0].kx, self.forward[0].ky]
            if isinstance(self.forward[0], conv.Conv)
            else self.forward[0].input)
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended

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
    load(Workflow, layers=root.cifar_caffe.layers)
    main(global_alpha=root.cifar_caffe.global_alpha,
         global_lambda=root.cifar_caffe.global_lambda,
         minibatch_maxsize=root.loader.minibatch_maxsize)
