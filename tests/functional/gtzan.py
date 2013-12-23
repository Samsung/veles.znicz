#!/usr/bin/python3.3 -O
"""
Created on Dec 9, 2013

File for GTZAN dataset.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import logging
import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import argparse
import numpy
import pickle
import re
import time
import traceback

import all2all
import config
import decision
import error
import evaluator
import gd
import loader
import opencl
import plotters
import rnd
import workflow
import formats


class Loader(loader.Loader):
    """Loads GTZAN dataset.
    """
    def __init__(self, pickle_fnme="", minibatch_max_size=100,
                 minibatches_in_epoch=1000, window_size=100, rnd=rnd.default2):
        super(Loader, self).__init__(minibatch_max_size=minibatch_max_size,
                                     rnd=rnd)
        self.pickle_fnme = pickle_fnme
        self.minibatches_in_epoch = minibatches_in_epoch
        self.data = None
        self.window_size = window_size
        self.features = ["Energy", "Centroid", "Flux", "Rolloff",
                         "ZeroCrossings", "CRP"]
        #                "MainBeat", "MainBeatStdDev"
        self.features_shape = {"CRP": 12}
        self.norm_add = {}
        self.norm_mul = {}
        self.labels = {"blues": 0,
                       "country": 1,
                       "jazz": 2,
                       "pop": 3,
                       "rock": 4,
                       "classical": 5,
                       "disco": 6,
                       "hiphop": 7,
                       "metal": 8,
                       "reggae": 9}
        self.exports = ["features", "labels", "norm_add", "norm_mul"]

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
                    self.log().error("%d NaNs occured for feature %s "
                        "at index %d in file %s" % (n, k,
                        numpy.isnan(vles).argmax(), fnme))
                sh = self.features_shape.get(k, 1)
                if sh != 1:
                    vles = vles.reshape(vles.size // sh, sh)
                features[k]["value"] = formats.max_type(vles)
                vles = features[k]["value"]
                nn = min(len(vles), nn)
                if nn < self.window_size:
                    raise error.ErrBadFormat("window_size=%d is too large "
                        "for feature %s with size %d in file %s" % (
                        self.window_size, k, nn, fnme))
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
        self.class_samples[2] = (self.minibatch_maxsize[0] *
                                 self.minibatches_in_epoch)

    def create_minibatches(self):
        nn = 0
        for k in self.features:
            nn += self.norm_add[k].size

        self.minibatch_data.reset()
        sh = [self.minibatch_maxsize[0], nn * self.window_size]
        self.minibatch_data.v = numpy.zeros(sh,
            dtype=config.dtypes[config.c_dtype])

        self.minibatch_target.reset()

        self.minibatch_labels.reset()
        sh = [self.minibatch_maxsize[0]]
        self.minibatch_labels.v = numpy.zeros(sh,
            dtype=numpy.int8)

        self.minibatch_indexes.reset()
        sh = [self.minibatch_maxsize[0]]
        self.minibatch_indexes.v = numpy.zeros(sh,
            dtype=config.itypes[config.get_itype_from_size(
                                len(self.data["files"]))])

    def shuffle_validation_train(self):
        pass

    def shuffle_train(self):
        pass

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
            self.minibatch_labels.v[i] = self.file_labels[idxs[i]]
            v = files[fnme]
            features = v["features"]
            limit = v["limit"]
            offs = rand.randint(limit)
            offs2 = offs + self.window_size
            j = 0
            for k in self.features:
                jj = j + self.norm_add[k].size * self.window_size
                self.minibatch_data.v[i, j:jj] = features[k]["value"][
                                                            offs:offs2].ravel()
                j = jj


class Workflow(workflow.NNWorkflow):
    """Sample workflow for MNIST dataset.
    """
    def __init__(self, layers=None, device=None):
        super(Workflow, self).__init__(device=device)

        self.rpt.link_from(self.start_point)

        self.loader = Loader()
        self.loader.link_from(self.rpt)

        # Add forward units
        self.forward.clear()
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh([layers[i]], device=device)
            else:
                aa = all2all.All2AllSoftmax([layers[i]], device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_idx = self.forward[-1].max_idx
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(snapshot_prefix="gtzan")
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.minibatch_max_err_y_sum = self.ev.max_err_y_sum
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(list(None for i in range(0, len(self.forward))))
        self.gd[-1] = gd.GDSM(device=device)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(device=device, weights_transposed=False)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
        self.rpt.link_from(self.gd[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(0, 3):
            self.plt.append(plotters.SimplePlotter(figure_label="num errors",
                                                   plot_style=styles[i]))
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if not i else self.plt[-2])
            self.plt[-1].gate_block = (self.decision.epoch_ended if not i
                                       else [1])
            self.plt[-1].gate_block_not = [1]
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Confusion matrix plotter
        self.plt_mx = []
        for i in range(0, len(self.decision.confusion_matrixes)):
            self.plt_mx.append(plotters.MatrixPlotter(
                figure_label=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision if not i
                                      else self.plt_mx[-2])
            self.plt_mx[-1].gate_block = (self.decision.epoch_ended if not i
                                          else [1])
            self.plt_mx[-1].gate_block_not = [1]
        # err_y plotter
        self.plt_err_y = []
        for i in range(0, 3):
            self.plt_err_y.append(plotters.SimplePlotter(
                figure_label="Last layer max gradient sum",
                plot_style=styles[i]))
            self.plt_err_y[-1].input = self.decision.max_err_y_sums
            self.plt_err_y[-1].input_field = i
            self.plt_err_y[-1].link_from(self.decision if not i
                                         else self.plt_err_y[-2])
            self.plt_err_y[-1].gate_block = (self.decision.epoch_ended if not i
                                             else [1])
            self.plt_err_y[-1].gate_block_not = [1]
        self.plt_err_y[0].clear_plot = True
        self.plt_err_y[-1].redraw_plot = True

    def initialize(self, device, args):
        self.loader.pickle_fnme = args.pickle_fnme
        self.loader.minibatch_maxsize[0] = args.minibatch_size
        self.loader.window_size = args.window_size
        self.decision.snapshot_prefix = args.snapshot_prefix
        for gd in self.gd:
            gd.global_alpha = args.global_alpha
            gd.global_lambda = args.global_lambda
        super(Workflow, self).initialize(device=device)
        return self.start_point.initialize_dependent()


def main():
    # if __debug__:
    #    logging.basicConfig(level=logging.DEBUG)
    # else:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, default="",
        help="Snapshot with trained network (default empty)")
    parser.add_argument("-export", type=bool,
        help="Export trained network to C (default False)",
        default=False)
    pickle_fnme = "%s/music/GTZAN/gtzan.pickle" % (config.test_dataset_root)
    parser.add_argument("-pickle_fnme", type=str,
        help="Pickle with input data (default %s)." % (pickle_fnme),
        default=pickle_fnme)
    parser.add_argument("-snapshot_prefix", type=str, required=True,
        help="Snapshot prefix (Ex.: gtzan_1000_500)")
    parser.add_argument("-layers", type=str, required=True,
        help="NN layer sizes, separated by any separator (Ex.: 1000_500_10)")
    parser.add_argument("-minibatch_size", type=int,
        help="Minibatch size (default 108)", default=108)
    parser.add_argument("-global_alpha", type=float,
        help="Global Alpha (default 0.01)", default=0.01)
    parser.add_argument("-global_lambda", type=float,
        help="Global Lambda (default 0.00005)", default=0.00005)
    parser.add_argument("-window_size", type=int,
        help="Window size (default 100)", default=100)
    args = parser.parse_args()

    s_layers = re.split("\D+", args.layers)
    layers = []
    for s in s_layers:
        layers.append(int(s))
    logging.info("Will train NN with layers: %s" % (" ".join(
                                        str(x) for x in layers)))

    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir),
                                    numpy.int32, 1024))
    rnd.default2.seed(numpy.fromfile("%s/seed2" % (this_dir),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    device = opencl.Device()
    try:
        fin = open(args.snapshot, "rb")
        w = pickle.load(fin)
        fin.close()
        if args.export:
            tm = time.localtime()
            s = "%d.%02d.%02d_%02d.%02d.%02d" % (
                tm.tm_year, tm.tm_mon, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec)
            fnme = "%s/channels_workflow_%s" % (config.snapshot_dir, s)
            try:
                w.export(fnme)
                logging.info("Exported successfully to %s.tar.gz" % (fnme))
            except:
                a, b, c = sys.exc_info()
                traceback.print_exception(a, b, c)
                logging.error("Error while exporting.")
            sys.exit(0)
    except IOError:
        w = Workflow(layers=layers, device=device)
    w.initialize(device=device, args=args)
    logging.info("norm_add: %s" % (str(w.loader.norm_add)))
    logging.info("norm_mul: %s" % (str(w.loader.norm_mul)))
    w.run()

    plotters.Graphics().wait_finish()
    logging.debug("End of job")


if __name__ == "__main__":
    main()
    if config.plotters_disabled:
        os._exit(0)
    sys.exit(0)
