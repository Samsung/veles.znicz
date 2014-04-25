#!/usr/bin/python3.3 -O
"""
Created on Dec 9, 2013

File for GTZAN dataset.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import pickle
import re

from veles.config import root, get
import veles.error as error
import veles.formats as formats
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.plotting_units as plotting_units
import veles.rnd as rnd
import veles.znicz.nn_units as nn_units
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.loader as loader

root.gtzan.labels = get(root.gtzan.labels, {"blues": 0,
                                            "country": 1,
                                            "jazz": 2,
                                            "pop": 3,
                                            "rock": 4,
                                            "classical": 5,
                                            "disco": 6,
                                            "hiphop": 7,
                                            "metal": 8,
                                            "reggae": 9})
root.gtzan.features_shape = get(root.gtzan.features_shape, {"CRP": 12})

root.defaults = {"decision": {"fail_iterations": 100,
                              "snapshot_prefix": "gtzan"},
                 "gtzan": {"exports":
                           ["features", "labels", "norm_add", "norm_mul"],
                           "features":
                           ["Energy", "Centroid", "Flux", "Rolloff",
                            "ZeroCrossings", "CRP"],
                           "global_alpha": 0.01,
                           "global_lambda": 0.00005,
                           "layers": [100, 500, 10],
                           "minibatch_maxsize": 108,
                           "minibatches_in_epoch": 1000,
                           "pickle_fnme":
                           os.path.join(root.common.test_dataset_root,
                                        "music/GTZAN/gtzan.pickle"),
                           "snapshot": "",
                           "window_size": 100}}


class Loader(loader.Loader):
    """Loads GTZAN dataset.
    """
    def __init__(self, workflow, **kwargs):
        pickle_fnme = kwargs.get("pickle_fnme", "")
        minibatch_max_size = kwargs.get("minibatch_max_size", 100)
        minibatches_in_epoch = kwargs.get("minibatches_in_epoch", 1000)
        window_size = kwargs.get("window_size", 100)
        rnd_ = kwargs.get("rnd", rnd.default2)
        labels = kwargs.get("labels", {"blues": 0,
                                       "country": 1,
                                       "jazz": 2,
                                       "pop": 3,
                                       "rock": 4,
                                       "classical": 5,
                                       "disco": 6,
                                       "hiphop": 7,
                                       "metal": 8,
                                       "reggae": 9})
        features = kwargs.get("features",
                              ["Energy", "Centroid", "Flux", "Rolloff",
                               "ZeroCrossings", "CRP"])
        exports = kwargs.get("exports", ["features", "labels", "norm_add",
                                         "norm_mul"])
        features_shape = kwargs.get("features_shape", {"CRP": 12})
        kwargs["pickle_fnme"] = pickle_fnme
        kwargs["minibatch_max_size"] = minibatch_max_size
        kwargs["minibatches_in_epoch"] = minibatches_in_epoch
        kwargs["window_size"] = window_size
        kwargs["rnd"] = rnd_
        kwargs["labels"] = labels
        kwargs["features"] = features
        kwargs["exports"] = exports
        kwargs["features_shape"] = features_shape
        super(Loader, self).__init__(workflow, **kwargs)
        self.pickle_fnme = pickle_fnme
        self.minibatches_in_epoch = minibatches_in_epoch
        self.data = None
        self.window_size = window_size
        self.features = features
        #                "MainBeat", "MainBeatStdDev"
        self.features_shape = features_shape
        self.norm_add = {}
        self.norm_mul = {}
        self.labels = labels
        self.exports = exports

    def __getstate__(self):
        state = super(Loader, self).__getstate__()
        state["data"] = None
        return state

    def load_data(self):
        fin = open(self.pickle_fnme, "rb")
        self.data = pickle.load(fin)
        fin.close()

        self.fnmes = list(self.data["files"].keys())
        self.fnmes.sort()
        self.file_labels = numpy.zeros(len(self.fnmes), dtype=numpy.int8)
        lbl_re = re.compile("/(\w+)\.\w+\.\w+$")
        for i, fnme in enumerate(self.fnmes):
            res = lbl_re.search(fnme)
            self.file_labels[i] = self.labels[res.group(1)]

        sums = {}
        counts = {}
        mins = {}
        maxs = {}
        train = self.data["files"]
        was_nans = False
        for fnme in sorted(train.keys()):
            v = train[fnme]
            features = v["features"]
            nn = 2000000000
            for k in self.features:
                vles = features[k]["value"]
                n = numpy.count_nonzero(numpy.isnan(vles))
                if n:
                    was_nans = True
                    self.error(
                        "%d NaNs occured for feature %s "
                        "at index %d in file %s" %
                        (n, k, numpy.isnan(vles).argmax(), fnme))
                sh = self.features_shape.get(k, 1)
                if sh != 1:
                    vles = vles.reshape(vles.size // sh, sh)
                features[k]["value"] = formats.max_type(vles)
                vles = features[k]["value"]
                nn = min(len(vles), nn)
                if nn < self.window_size:
                    raise error.ErrBadFormat(
                        "window_size=%d is too large "
                        "for feature %s with size %d in file %s" %
                        (self.window_size, k, nn, fnme))
            v["limit"] = nn - self.window_size + 1
            for k in self.features:
                vles = features[k]["value"][:nn]
                if k in sums.keys():
                    sums[k] += vles.sum(axis=0)
                    counts[k] += vles.shape[0]
                    mins[k] = numpy.minimum(vles.min(axis=0), mins[k])
                    maxs[k] = numpy.maximum(vles.max(axis=0), maxs[k])
                else:
                    sums[k] = vles.sum(axis=0)
                    counts[k] = vles.shape[0]
                    mins[k] = vles.min(axis=0)
                    maxs[k] = vles.max(axis=0)
        if was_nans:
            raise error.ErrBadFormat("There were NaNs.")

        for k in self.features:
            mean = sums[k] / counts[k]
            self.norm_add[k] = -mean
            df = max(mean - mins[k], maxs[k] - mean)
            self.norm_mul[k] = 1.0 / df if df else 0

        for v in self.data["files"].values():
            features = v["features"]
            for k in self.features:
                vles = features[k]["value"]
                vles += self.norm_add[k]
                vles *= self.norm_mul[k]

        self.class_samples[0] = 0
        self.class_samples[1] = 0
        self.class_samples[2] = (self.minibatch_maxsize *
                                 self.minibatches_in_epoch)

    def create_minibatches(self):
        nn = 0
        for k in self.features:
            nn += self.norm_add[k].size

        self.minibatch_data.reset()
        sh = [self.minibatch_maxsize, nn * self.window_size]
        self.minibatch_data.v = numpy.zeros(
            sh, dtype=opencl_types.dtypes[root.common.precision_type])

        self.minibatch_target.reset()

        self.minibatch_labels.reset()
        sh = [self.minibatch_maxsize]
        self.minibatch_labels.v = numpy.zeros(
            sh, dtype=numpy.int8)

        self.minibatch_indexes.reset()
        sh = [self.minibatch_maxsize]
        self.minibatch_indexes.v = numpy.zeros(
            sh, dtype=opencl_types.itypes[
                opencl_types.get_itype_from_size(len(self.data["files"]))])

    def shuffle(self):
        pass

    def fill_minibatch(self):
        super(Loader, self).fill_minibatch()

        minibatch_size = self.minibatch_size[0]

        rand = self.rnd[0]
        n = len(self.fnmes)
        idxs = self.minibatch_indexes.v
        idxs[:] = rand.randint(0, n, minibatch_size)[:]
        files = self.data["files"]
        for i in range(minibatch_size):
            fnme = self.fnmes[idxs[i]]
            self.minibatch_labels[i] = self.file_labels[idxs[i]]
            v = files[fnme]
            features = v["features"]
            limit = v["limit"]
            offs = rand.randint(limit)
            offs2 = offs + self.window_size
            j = 0
            for k in self.features:
                jj = j + self.norm_add[k].size * self.window_size
                self.minibatch_data.v[
                    i, j:jj] = features[k]["value"][offs:offs2].ravel()
                j = jj


class Workflow(nn_units.NNWorkflow):
    """Sample workflow for MNIST dataset.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(
            self, labels=root.gtzan.labels, exports=root.gtzan.exports,
            pickle_fnme=root.gtzan.pickle_fnme,
            minibatch_maxsize=root.gtzan.minibatch_maxsize,
            minibatches_in_epoch=root.gtzan.minibatches_in_epoch,
            window_size=root.gtzan.window_size, features=root.gtzan.features,
            features_shape=root.gtzan.features_shape)
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
                self.fwds[i].input = self.fwds[i - 1].output
            else:
                self.fwds[i].link_from(self.loader)
                self.fwds[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.evaluator = evaluator.EvaluatorSoftmax(self, device=device)
        self.evaluator.link_from(self.fwds[-1])
        self.evaluator.y = self.fwds[-1].output
        self.evaluator.batch_size = self.loader.minibatch_size
        self.evaluator.labels = self.loader.minibatch_labels
        self.evaluator.max_idx = self.fwds[-1].max_idx
        self.evaluator.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(
            self, snapshot_prefix=root.decision.snapshot_prefix,
            fail_iterations=root.decision.fail_iterations)
        self.decision.link_from(self.evaluator)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.link_attrs(self.loader, "no_more_minibatches_left")
        self.decision.minibatch_n_err = self.evaluator.n_err
        self.decision.minibatch_confusion_matrix = self.evaluator.confusion_matrix
        self.decision.minibatch_max_err_y_sum = self.evaluator.max_err_y_sum
        self.decision.class_samples = self.loader.class_samples

        # Add gradient descent units
        del self.gds[:]
        self.gds.extend(list(None for i in range(0, len(self.fwds))))
        self.gds[-1] = gd.GDSM(self, device=device)
        self.gds[-1].link_from(self.decision)
        self.gds[-1].err_y = self.evaluator.err_y
        self.gds[-1].y = self.fwds[-1].output
        self.gds[-1].h = self.fwds[-1].input
        self.gds[-1].weights = self.fwds[-1].weights
        self.gds[-1].bias = self.fwds[-1].bias
        self.gds[-1].gate_skip = self.decision.gd_skip
        self.gds[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.fwds) - 2, -1, -1):
            self.gds[i] = gd.GDTanh(self, device=device,
                                   weights_transposed=False)
            self.gds[i].link_from(self.gds[i + 1])
            self.gds[i].err_y = self.gds[i + 1].err_h
            self.gds[i].y = self.fwds[i].output
            self.gds[i].h = self.fwds[i].input
            self.gds[i].weights = self.fwds[i].weights
            self.gds[i].bias = self.fwds[i].bias
            self.gds[i].gate_skip = self.decision.gd_skip
            self.gds[i].batch_size = self.loader.minibatch_size
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
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else self.plt[-2])
            self.plt[-1].gate_block = (~self.decision.epoch_ended if not i
                                       else Bool(False))
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(0, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotting_units.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision if not i
                                      else self.plt_mx[-2])
            self.plt_mx[-1].gate_block = (~self.decision.epoch_ended if not i
                                          else Bool(False))
        # err_y plotter
        self.plt_err_y = []
        for i in range(0, 3):
            self.plt_err_y.append(plotting_units.AccumulatingPlotter(
                self, name="Last layer max gradient sum",
                plot_style=styles[i]))
            self.plt_err_y[-1].input = self.decision.max_err_y_sums
            self.plt_err_y[-1].input_field = i
            self.plt_err_y[-1].link_from(self.decision if not i
                                         else self.plt_err_y[-2])
            self.plt_err_y[-1].gate_block = (~self.decision.epoch_ended if not
                                             i else Bool(False))
        self.plt_err_y[0].clear_plot = True
        self.plt_err_y[-1].redraw_plot = True

    def initialize(self, global_alpha, global_lambda, device):
        super(Workflow, self).initialize(global_alpha=global_alpha,
                                         global_lambda=global_lambda,
                                         device=device)


def run(load, main):
    w, _ = load(Workflow, layers=root.gtzan.layers)
    logging.info("norm_add: %s" % (str(w.loader.norm_add)))
    logging.info("norm_mul: %s" % (str(w.loader.norm_mul)))
    main(global_alpha=root.gtzan.global_alpha,
         global_lambda=root.gtzan.global_lambda)
