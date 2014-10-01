#!/usr/bin/python3 -O
"""
Created on June 29, 2013

File for kanji recognition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import pickle
import re
import sys
from zope.interface import implementer

from veles.config import root
import veles.error as error
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.prng as rnd
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader
import veles.znicz.nn_plotting_units as nn_plotting_units
from veles.znicz.nn_units import NNSnapshotter


train_path = os.path.join(root.common.test_dataset_root, "kanji/train")

root.defaults = {
    "decision": {"fail_iterations": 1000,
                 "store_samples_mse": True},
    "snapshotter": {"prefix": "kanji"},
    "loader": {"minibatch_size": 5103,
               "validation_ratio": 0.15},
    "weights_plotter": {"limit": 16},
    "kanji": {"learning_rate": 0.0000001,
              "weights_decay": 0.00005,
              "layers": [5103, 2889, 24 * 24],
              "data_paths":
              {"target": os.path.join(root.common.test_dataset_root,
                                      "kanji/target/targets.%d.pickle" %
                                      (sys.version_info[0])),
               "train": train_path},
              "index_map": os.path.join(train_path, "index_map.%d.pickle" %
                                        (sys.version_info[0]))}}


@implementer(loader.ILoader)
class KanjiLoader(loader.LoaderMSE):
    """Loads dataset.
    """
    def __init__(self, workflow, **kwargs):
        self.train_path = kwargs["train_path"]
        self.target_path = kwargs["target_path"]
        super(KanjiLoader, self).__init__(workflow, **kwargs)
        self.class_targets = formats.Vector()

    def __getstate__(self):
        state = super(KanjiLoader, self).__getstate__()
        state["index_map"] = None
        return state

    def load_data(self):
        """Load the data here.

        Should be filled here:
            class_lengths[].
        """
        fin = open(root.kanji.index_map, "rb")
        self.index_map = pickle.load(fin)
        fin.close()

        fin = open(os.path.join(self.train_path, self.index_map[0]), "rb")
        self.first_sample = pickle.load(fin)["data"]
        fin.close()

        fin = open(self.target_path, "rb")
        targets = pickle.load(fin)
        fin.close()
        self.class_targets.reset()
        sh = [len(targets)]
        sh.extend(targets[0].shape)
        self.class_targets.mem = numpy.empty(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])
        for i, target in enumerate(targets):
            self.class_targets[i] = target

        self.class_lengths[0] = 0
        self.class_lengths[1] = 0
        self.class_lengths[2] = len(self.index_map)

        self.original_labels = numpy.empty(len(self.index_map),
                                           dtype=numpy.int32)
        lbl_re = re.compile("^(\d+)_\d+/(\d+)\.\d\.pickle$")
        for i, fnme in enumerate(self.index_map):
            res = lbl_re.search(fnme)
            if res is None:
                raise error.BadFormatError("Incorrectly formatted filename "
                                           "found: %s" % (fnme))
            lbl = int(res.group(1))
            self.original_labels[i] = lbl
            idx = int(res.group(2))
            if idx != i:
                raise error.BadFormatError("Incorrect sample index extracted "
                                           "from filename: %s " % (fnme))

        self.info("Found %d samples. Extracting 15%% for validation..." % (
            len(self.index_map)))
        self.extract_validation_from_train(rnd.get(2))
        self.info("Extracted, resulting datasets are: [%s]" % (
            ", ".join(str(x) for x in self.class_lengths)))

    def create_minibatches(self):
        """Allocate arrays for minibatch_data etc. here.
        """
        sh = [self.max_minibatch_size]
        sh.extend(self.first_sample.shape)
        self.minibatch_data.mem = numpy.zeros(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])

        sh = [self.max_minibatch_size]
        sh.extend((self.class_targets[0].size,))
        self.minibatch_targets.mem = numpy.zeros(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])

        sh = [self.max_minibatch_size]
        self.minibatch_labels.mem = numpy.zeros(sh, dtype=numpy.int32)

        self.minibatch_indices.mem = numpy.zeros(len(self.index_map),
                                                 dtype=numpy.int32)

    def fill_minibatch(self):
        """Fill minibatch data labels and indexes according to current shuffle.
        """
        idxs = self.minibatch_indices.mem
        for i, ii in enumerate(idxs[:self.minibatch_size]):
            fnme = "%s/%s" % (self.train_path, self.index_map[ii])
            fin = open(fnme, "rb")
            sample = pickle.load(fin)
            data = sample["data"]
            lbl = sample["lbl"]
            fin.close()
            self.minibatch_data[i] = data
            self.minibatch_labels[i] = lbl
            self.minibatch_targets[i] = self.class_targets[lbl].reshape(
                self.minibatch_targets[i].shape)


class KanjiWorkflow(nn_units.NNWorkflow):
    """Workflow for training network which will be able to recognize
    drawn kanji characters; training done using only TrueType fonts;
    1023 classes to recognize, 3.6 million 32x32 images dataset size.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["name"] = kwargs.get("name", "Kanji")
        super(KanjiWorkflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = KanjiLoader(
            self, validation_ratio=root.loader.validation_ratio,
            train_path=root.kanji.data_paths.train,
            target_path=root.kanji.data_paths.target)
        self.loader.link_from(self.repeater)

        # Add fwds units
        del self.fwds[:]
        for i in range(0, len(layers)):
            aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
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

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorMSE(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.link_attrs(self.loader,
                                  ("batch_size", "minibatch_size"),
                                  ("max_samples_per_epoch", "total_samples"),
                                  ("target", "minibatch_targets"),
                                  ("labels", "minibatch_labels"),
                                  "class_targets")
        self.evaluator.link_attrs(self.fwds[-1], "output")

        # Add decision unit
        self.decision = decision.DecisionMSE(
            self, fail_iterations=root.decision.fail_iterations,
            store_samples_mse=root.decision.store_samples_mse)
        self.decision.link_from(self.evaluator)
        self.decision.link_attrs(self.loader,
                                 "minibatch_class", "minibatch_size",
                                 "last_minibatch", "class_lengths",
                                 "epoch_ended", "epoch_number",
                                 "minibatch_offset", "minibatch_size")
        self.decision.link_attrs(
            self.evaluator,
            ("minibatch_n_err", "n_err"),
            ("minibatch_metrics", "metrics"),
            ("minibatch_mse", "mse"))

        self.snapshotter = NNSnapshotter(self, prefix=root.snapshotter.prefix,
                                         directory=root.common.snapshot_dir)
        self.snapshotter.link_from(self.decision)
        self.snapshotter.link_attrs(self.decision,
                                    ("suffix", "snapshot_suffix"))
        self.snapshotter.gate_skip = \
            (~self.loader.epoch_ended | ~self.decision.improved)

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(None for i in range(0, len(self.fwds)))
        self.gds[-1] = gd.GDTanh(self, device=device)
        self.gds[-1].link_from(self.snapshotter)
        self.gds[-1].link_attrs(self.fwds[-1], "output", "input",
                                "weights", "bias")
        self.gds[-1].link_attrs(self.evaluator, "err_output")
        self.gds[-1].link_attrs(self.loader, ("batch_size", "minibatch_size"))
        self.gds[-1].gate_skip = self.decision.gd_skip
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].link_attrs(self.fwds[i],
                                   "output", "input",
                                   "weights", "bias")
            self.gds[i].link_attrs(self.loader, ("batch_size",
                                                 "minibatch_size"))
            self.gds[i].link_attrs(self.gds[i + 1],
                                   ("err_output", "err_input"))
            self.gds[i].gate_skip = self.decision.gd_skip
        self.repeater.link_from(self.gds[0])

        self.end_point.link_from(self.gds[0])
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # MSE plotter
        self.plt = []
        styles = ["", "", "k-"]  # ["r-", "b-", "k-"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt[-1].link_attrs(self.decision, ("input", "epoch_metrics"))
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        self.plt[0].clear_plot = True

        # Weights plotter
        self.plt_mx = nn_plotting_units.Weights2D(
            self, name="First Layer Weights",
            limit=root.weights_plotter.limit)
        self.plt_mx.link_attrs(self.gds[0], ("input", "weights"))
        self.plt_mx.link_attrs(self.fwds[0], ("get_shape_from", "input"))
        self.plt_mx.input_field = "mem"
        self.plt_mx.link_from(self.decision)
        self.plt_mx.gate_block = ~self.decision.epoch_ended

        # Max plotter
        self.plt_max = []
        styles = ["", "", "k--"]  # ["r--", "b--", "k--"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt_max.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_max[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_max[-1].input_field = i
            self.plt_max[-1].input_offset = 1
            self.plt_max[-1].link_from(self.decision)
            self.plt_max[-1].gate_block = ~self.decision.epoch_ended
        # Min plotter
        self.plt_min = []
        styles = ["", "", "k:"]  # ["r:", "b:", "k:"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt_min.append(plotting_units.AccumulatingPlotter(
                self, name="mse", plot_style=styles[i]))
            self.plt_min[-1].link_attrs(self.decision,
                                        ("input", "epoch_metrics"))
            self.plt_min[-1].input_field = i
            self.plt_min[-1].input_offset = 2
            self.plt_min[-1].link_from(self.decision)
            self.plt_min[-1].gate_block = ~self.decision.epoch_ended
        self.plt_min[-1].redraw_plot = True
        # Error plotter
        self.plt_n_err = []
        styles = ["", "", "k-"]  # ["r-", "b-", "k-"]
        for i in range(len(styles)):
            if not len(styles[i]):
                continue
            self.plt_n_err.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i]))
            self.plt_n_err[-1].link_attrs(self.decision,
                                          ("input", "epoch_n_err_pt"))
            self.plt_n_err[-1].input_field = i
            self.plt_n_err[-1].link_from(self.decision)
            self.plt_n_err[-1].gate_block = ~self.decision.epoch_ended
        self.plt_n_err[0].clear_plot = True
        self.plt_n_err[-1].redraw_plot = True

        # Image plotter
        self.plt_img = plotting_units.ImagePlotter(self, name="output sample")
        self.plt_img.inputs.append(self.fwds[-1].output)
        self.plt_img.input_fields.append(0)
        self.plt_img.inputs.append(self.fwds[0].input)
        self.plt_img.input_fields.append(0)
        self.plt_img.link_from(self.decision)
        self.plt_img.gate_skip = ~self.decision.epoch_ended

    def initialize(self, device, learning_rate, weights_decay, minibatch_size,
                   weights, bias, **kwargs):
        super(KanjiWorkflow, self).initialize(learning_rate=learning_rate,
                                              weights_decay=weights_decay,
                                              minibatch_size=minibatch_size,
                                              device=device)
        if weights is not None:
            for i, fwds in enumerate(self.fwds):
                fwds.weights.map_invalidate()
                fwds.weights.mem[:] = weights[i][:]
        if bias is not None:
            for i, fwds in enumerate(self.fwds):
                fwds.bias.map_invalidate()
                fwds.bias.mem[:] = bias[i][:]


def run(load, main):
    weights = None
    bias = None
    w, snapshot = load(KanjiWorkflow, layers=root.kanji.layers)
    if snapshot:
        if type(w) == tuple:
            logging.info("Will load weights")
            weights = w[0]
            bias = w[1]
        else:
            logging.info("Will load workflow")
            logging.info("Weights and bias ranges per layer are:")
            for fwds in w.fwds:
                logging.info("%f %f %f %f" % (
                    fwds.weights.mem.min(), fwds.weights.mem.max(),
                    fwds.bias.mem.min(), fwds.bias.mem.max()))
            w.decision.improved <<= True
    main(learning_rate=root.kanji.learning_rate,
         weights_decay=root.kanji.weights_decay,
         minibatch_size=root.loader.minibatch_size,
         weights=weights, bias=bias)
