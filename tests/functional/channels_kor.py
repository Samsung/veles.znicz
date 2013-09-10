#!/usr/bin/python3.3 -O
"""
Created on Sep 2, 2013

File for korean channels recognition.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import os
import logging


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s" % (this_dir))
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import units
import formats
import numpy
import config
import rnd
import opencl
import plotters
import glob
import pickle
import image
import tv_channel_plotter
import loader
import decision
import image_saver
import all2all
import evaluator
import gd
import glymur
import workflow
import scipy.io


class Loader(loader.FullBatchLoader):
    """Loads channels.
    """
    def __init__(self, minibatch_max_size=100, rnd=rnd.default,
                 channels_dir="%s/channels/korean_960_540/by_number" % (
                                                config.test_dataset_root),
                 rect=(160, 80), grayscale=False):
        super(Loader, self).__init__(minibatch_max_size=minibatch_max_size,
                                     rnd=rnd)
        self.conf_ = None
        self.channels_dir = channels_dir
        self.rect = rect
        self.grayscale = grayscale
        self.channel_map = None
        self.pos = {}
        self.sz = [0, 0]
        self.attributes_for_cached_data = [
            "channels_dir", "rect", "channel_map", "pos", "sz",
            "class_samples", "grayscale"]

    def from_jp2(self, fnme):
        j2 = glymur.Jp2k(fnme)
        a2 = j2.read()  # returns interleaved yuv444
        if self.grayscale:
            a = numpy.empty([a2.shape[0], a2.shape[1], 1],
                dtype=config.dtypes[config.dtype])
            a[:, :, 0:1] = a2[:, :, 0:1]
            a = formats.reshape(a, [a2.shape[0], a2.shape[1]])
        else:
            # transform to different yuv planes
            a = numpy.empty([3, a2.shape[0], a2.shape[1]],
                dtype=config.dtypes[config.dtype])
            a[0:1, :, :].reshape(
                a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 0:1]
            a[1:2, :, :].reshape(
                a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 1:2]
            a[2:3, :, :].reshape(
                a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 2:3]
        return a

    def load_data(self):
        cached_data_fnme = "%s/%s_%s.pickle" % (
            config.cache_dir, __file__, self.__class__.__name__)
        self.log().info("Will try to load previously cached data from "
                        "%s" % (cached_data_fnme))
        try:
            fin = open(cached_data_fnme, "rb")
            obj = pickle.load(fin)
            for k, v in obj.items():
                if type(v) == list:
                    o = self.__dict__[k]
                    if o == None:
                        o = []
                        self.__dict__[k] = o
                    o.clear()
                    o.extend(v)
                elif type(v) == dict:
                    o = self.__dict__[k]
                    if o == None:
                        o = {}
                        self.__dict__[k] = o
                    o.update(v)
                else:
                    self.__dict__[k] = v
            self.original_labels = pickle.load(fin)
            a = pickle.load(fin)
            sh = [self.original_labels.shape[0]]
            sh.extend(a.shape)
            self.original_data = numpy.zeros(sh,
                dtype=config.dtypes[config.dtype])
            self.original_data[0] = a
            for i in range(1, self.original_data.shape[0]):
                a = pickle.load(fin)
                self.original_data[i] = a
            fin.close()
            self.log().info("Succeeded")
            fnme = "%s/ch.mat" % (config.cache_dir)
            self.log().info("Exporting to matlab file: %s" % (fnme))
            scipy.io.savemat(fnme, {"data": self.original_data,
                                    "labels": self.original_labels})
            self.log().info("Done")
            return
        except FileNotFoundError:
            self.log().info("Failed")

        self.log().info("Will load data from original jp2 files")
        try:
            add_path(self.channels_dir)
            self.conf_ = __import__("conf")
        except:
            self.log().error("Error while importing %s/config.py" % (
                                                    self.channels_dir))
            raise
        # Parse config
        pos = {}
        rpos = {}
        frame = self.conf_.frame
        self.channel_map = self.conf_.channel_map
        for conf in self.channel_map.values():
            if conf["type"] not in pos.keys():
                pos[conf["type"]] = frame.copy()
                rpos[conf["type"]] = [0, 0]
            pos[conf["type"]][0] = min(pos[conf["type"]][0], conf["pos"][0])
            pos[conf["type"]][1] = min(pos[conf["type"]][1], conf["pos"][1])
            rpos[conf["type"]][0] = max(rpos[conf["type"]][0],
                conf["pos"][0] + conf["size"][0])
            rpos[conf["type"]][1] = max(rpos[conf["type"]][1],
                conf["pos"][1] + conf["size"][1])

        self.log().info("Found rectangles:")
        sz = [0, 0]
        for k in pos.keys():
            sz[0] = max(sz[0], rpos[k][0] - pos[k][0])
            sz[1] = max(sz[1], rpos[k][1] - pos[k][1])
            self.log().info("%s: pos: (%d, %d)" % (k, pos[k][0], pos[k][1]))
        self.log().info("sz: (%d, %d)" % (sz[0], sz[1]))

        self.log().info("Adjusted rectangles:")
        sz[0] += 16
        sz[1] += 16
        for k in pos.keys():
            pos[k][0] -= (rpos[k][0] - pos[k][0] - sz[0]) >> 1
            pos[k][1] -= (rpos[k][1] - pos[k][1] - sz[1]) >> 1
            pos[k][0] = min(pos[k][0], frame[0] - sz[0])
            pos[k][1] = min(pos[k][1], frame[1] - sz[1])
            pos[k][0] = max(pos[k][0], 0)
            pos[k][1] = max(pos[k][1], 0)
            self.log().info("%s: pos: (%d, %d)" % (k, pos[k][0], pos[k][1]))
            # Calculate relative values
            pos[k][0] /= frame[0]
            pos[k][1] /= frame[1]
        sz[0] /= frame[0]
        sz[1] /= frame[1]

        self.pos.clear()
        self.pos.update(pos)
        self.sz.clear()
        self.sz.extend(sz)

        files = {}
        total_files = 0
        total_samples = 0
        dirs = [self.conf_.no_channel_dir]
        dirs.extend(self.channel_map.keys())
        dirs.sort()
        for dirnme in dirs:
            files[dirnme] = glob.glob("%s/%s/*.jp2" % (
                                self.channels_dir, dirnme))
            files[dirnme].sort()
            total_files += len(files[dirnme])
            # We will extract data from every corner.
            total_samples += len(files[dirnme]) * len(pos.keys())
        self.log().info("Found %d files" % (total_files))
        self.log().info("Together with negative set "
                        "will generate %d samples" % (total_samples))

        self.original_labels = numpy.zeros(total_samples,
            dtype=config.itypes[config.get_itype_from_size(len(dirs))])
        if self.grayscale:
            self.original_data = numpy.zeros([total_samples,
                self.rect[1], self.rect[0]], config.dtypes[config.dtype])
        else:
            self.original_data = numpy.zeros([total_samples, 3,
                self.rect[1], self.rect[0]], config.dtypes[config.dtype])
        i = 0
        n_files = 0
        for dirnme in dirs:
            self.log().info("Loading from %s" % (dirnme))
            for fnme in files[dirnme]:
                a = self.from_jp2(fnme)
                # Data from corners will form samples.
                for k in pos.keys():
                    if (dirnme in self.channel_map.keys() and
                        self.channel_map[dirnme]["type"] == k):
                        self.original_labels[i] = int(dirnme)
                    else:
                        self.original_labels[i] = 0

                    if self.grayscale:
                        x = numpy.rot90(a, 2)
                        left = pos[k][0] * x.shape[1]
                        top = pos[k][1] * x.shape[0]
                        width = sz[0] * x.shape[1]
                        height = sz[1] * x.shape[0]
                        x = x[top:top + height, left:left + width]
                        x = image.resize(x, self.rect[0], self.rect[1])
                        self.original_data[i] = x
                    else:
                        # Loop by color planes.
                        for j in range(0, a.shape[0]):
                            x = numpy.rot90(a[j], 2)
                            left = pos[k][0] * x.shape[1]
                            top = pos[k][1] * x.shape[0]
                            width = sz[0] * x.shape[1]
                            height = sz[1] * x.shape[0]
                            x = x[top:top + height, left:left + width]
                            x = image.resize(x, self.rect[0], self.rect[1])
                            self.original_data[i, j] = x

                    if self.grayscale:
                        formats.normalize(self.original_data[i])
                    else:
                        # Normalize Y and UV planes separately.
                        formats.normalize(self.original_data[i][0])
                        formats.normalize(self.original_data[i][1:])

                    i += 1
                n_files += 1
                self.log().info("Read %d files (%.2f%%)" % (n_files,
                                100.0 * n_files / total_files))

        self.class_samples[0] = 0
        self.class_samples[1] = 0
        self.class_samples[2] = self.original_data.shape[0]

        self.log().info("Saving loaded data for later faster load to "
                        "%s" % (cached_data_fnme))
        fout = open(cached_data_fnme, "wb")
        obj = {}
        for name in self.attributes_for_cached_data:
            obj[name] = self.__dict__[name]
        pickle.dump(obj, fout)
        pickle.dump(self.original_labels, fout)
        for a in self.original_data:
            pickle.dump(a, fout)
        fout.close()


class Workflow(workflow.NNWorkflow):
    """Workflow.
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

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(out_dirs=[
            "/data/veles/channels/korean_960_540/tmpimg/test",
            "/data/veles/channels/korean_960_540/tmpimg/validation",
            "/data/veles/channels/korean_960_540/tmpimg/train"], limit=100)
        self.image_saver.link_from(self.forward[-1])
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.forward[-1].output
        self.image_saver.max_idx = self.forward[-1].max_idx
        self.image_saver.indexes = self.loader.minibatch_indexes
        self.image_saver.labels = self.loader.minibatch_labels
        self.image_saver.minibatch_class = self.loader.minibatch_class
        self.image_saver.minibatch_size = self.loader.minibatch_size

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(device=device)
        self.ev.link_from(self.image_saver)
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_idx = self.forward[-1].max_idx
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(snapshot_prefix="channels_kor",
                                          use_dynamic_alpha=False)
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err_skipped
        self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        self.image_saver.gate_skip = self.decision.just_snapshotted
        self.image_saver.gate_skip_not = [1]
        self.image_saver.this_save_time = self.decision.snapshot_time

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(list(None for i in range(0, len(self.forward))))
        self.gd[-1] = gd.GDSM(device=device)
        #self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
        self.rpt.link_from(self.gd[0])

        self.end_point = units.EndPoint()
        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(2, 3):
            self.plt.append(plotters.SimplePlotter(figure_label="num errors",
                                                   plot_style=styles[i],
                                                   bounds=(0, 100)))
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_skip = self.decision.epoch_ended
            self.plt[-1].gate_skip_not = [1]
        self.plt[0].clear_plot = True
        # Weights plotter
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_w = plotters.Weights2D(figure_label="First Layer Weights",
                                        limit=16)
        self.plt_w.input = self.gd[0].weights
        self.plt_w.get_shape_from = self.forward[0].input
        self.plt_w.input_field = "v"
        self.plt_w.link_from(self.plt[-1])
        self.plt_w.gate_skip = self.decision.epoch_ended
        self.plt_w.gate_skip_not = [1]
        # Image plottter
        self.decision.vectors_to_sync[self.forward[0].input] = 1
        self.plt_i = plotters.Image(figure_label="Input")
        self.plt_i.inputs.append(self.decision)
        self.plt_i.input_fields.append("sample_input")
        self.plt_i.link_from(self.plt_w)
        self.plt_i.gate_skip = self.decision.epoch_ended
        self.plt_i.gate_skip_not = [1]
        # Confusion matrix plotter
        #"""
        self.plt_mx = []
        for i in range(2, 3):
            self.plt_mx.append(plotters.MatrixPlotter(
                figure_label=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.plt_i)
            self.plt_mx[-1].gate_skip = self.decision.epoch_ended
            self.plt_mx[-1].gate_skip_not = [1]
        #"""
        self.gd[-1].link_from(self.plt_mx[-1])

    def initialize(self, threshold, threshold_low,
                   global_alpha, global_lambda,
                   minibatch_maxsize, device):
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        self.ev.device = device
        self.ev.threshold = threshold
        self.ev.threshold_low = threshold_low
        for gd in self.gd:
            gd.device = device
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
        for forward in self.forward:
            forward.device = device
        return self.start_point.initialize_dependent()


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir),
                                    numpy.int32, 1024))
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    cl = opencl.DeviceList()
    device = cl.get_device()
    try:
        fin = open("%s/channels_kor.pickle" % (config.snapshot_dir), "rb")
        w = pickle.load(fin)
        fin.close()
    except IOError:
        w = Workflow(layers=[50, 28], device=device)
    w.initialize(threshold=1.0, threshold_low=1.0,
                 global_alpha=0.001, global_lambda=0.0,
                 minibatch_maxsize=54, device=device)
    w.run()
    plotters.Graphics().wait_finish()
    logging.info("End of job")


if __name__ == "__main__":
    main()
