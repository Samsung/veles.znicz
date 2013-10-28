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
import pickle
import image
import loader
import decision
import image_saver
import error
import all2all
import evaluator
import gd
import glymur
import workflow
import scipy.io
import scipy.misc
import thread_pool
import threading
import re


class Loader(loader.FullBatchLoader):
    """Loads channels.
    """
    def __init__(self, minibatch_max_size=100, rnd=rnd.default,
                 channels_dir="%s/channels/korean_960_540/train" % (
                                                config.test_dataset_root),
                 rect=(176, 96), grayscale=False):
        super(Loader, self).__init__(minibatch_max_size=minibatch_max_size,
                                     rnd=rnd)
        #: Top-level configuration from channels_dir/conf.py
        self.top_conf_ = None
        #: Configuration from channels_dir/subdirectory/conf.py
        self.subdir_conf_ = {}
        self.channels_dir = channels_dir
        self.rect = rect
        self.grayscale = grayscale
        self.w_neg = None  # workflow for finding the negative dataset
        self.find_negative = 0
        self.channel_map = None
        self.pos = {}
        self.sz = [0, 0]
        self.file_map = {}  # sample index to its file name map
        self.attributes_for_cached_data = [
            "channels_dir", "rect", "channel_map", "pos", "sz",
            "class_samples", "grayscale", "file_map"]
        self.exports = ["rect", "pos", "sz"]

    def from_jp2(self, fnme, rot):
        j2 = glymur.Jp2k(fnme)
        a2 = j2.read()  # returns interleaved yuv444
        if rot:
            a2 = numpy.rot90(a2, 2)
        if self.grayscale:
            a = numpy.empty([a2.shape[0], a2.shape[1], 1], dtype=numpy.uint8)
            a[:, :, 0:1] = a2[:, :, 0:1]
            a = formats.reshape(a, [a2.shape[0], a2.shape[1]])
        else:
            # transform to different yuv planes
            a = numpy.empty([3, a2.shape[0], a2.shape[1]], dtype=numpy.uint8)
            a[0:1, :, :].reshape(
                a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 0:1]
            a[1:2, :, :].reshape(
                a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 1:2]
            a[2:3, :, :].reshape(
                a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 2:3]
        return a

    def sample_rect(self, a, pos, sz):
        aa = numpy.empty_like(self.original_data[0])
        if self.grayscale:
            x = a
            left = int(numpy.round(pos[0] * x.shape[1]))
            top = int(numpy.round(pos[1] * x.shape[0]))
            width = int(numpy.round(sz[0] * x.shape[1]))
            height = int(numpy.round(sz[1] * x.shape[0]))
            x = x[top:top + height, left:left + width].ravel().copy().\
                reshape((height, width), order="C")
            x = image.resize(x, self.rect[0], self.rect[1])
            aa[:] = x[:]
        else:
            # Loop by color planes.
            for j in range(0, a.shape[0]):
                x = a[j]
                left = int(numpy.round(pos[0] * x.shape[1]))
                top = int(numpy.round(pos[1] * x.shape[0]))
                width = int(numpy.round(sz[0] * x.shape[1]))
                height = int(numpy.round(sz[1] * x.shape[0]))
                x = x[top:top + height, left:left + width].ravel().copy().\
                    reshape((height, width), order="C")
                x = image.resize(x, self.rect[0], self.rect[1])
                aa[j] = x

        if self.grayscale:
            formats.normalize(aa)
        else:
            # Normalize Y and UV planes separately.
            formats.normalize(aa[0])
            formats.normalize(aa[1:])

        return aa

    def from_jp2_async(self, fnme, pos, sz,
                       data_lock, i_sample, lbl, n_files, total_files,
                       negative_data, negative_file_map, rand):
        """Loads, crops and normalizes image in the parallel thread.
        """
        a = self.from_jp2(fnme, fnme.find("norotate") < 0)

        self.original_labels[i_sample] = lbl
        self.original_data[i_sample] = self.sample_rect(a, pos, sz)
        self.file_map[i_sample] = fnme

        # Collect negative dataset from positive samples only
        if lbl and self.w_neg != None and self.find_negative > 0:
            negative_data[i_sample] = []
            negative_file_map[i_sample] = fnme
            # Sample pictures at random positions
            for i in range(self.find_negative):
                t = rand.randint(2)
                if t == 0:
                    # Sample vertical line
                    p = [pos[0] + (1 if pos[0] < 0.5 else -1) * sz[0],
                         rand.rand() * (1.0 - sz[1])]
                elif t == 1:
                    # Sample horizontal line
                    p = [rand.rand() * (1.0 - sz[0]),
                         pos[1] + (1 if pos[1] < 0.5 else -1) * sz[1]]
                else:
                    continue
                negative_data[i_sample].append(self.sample_rect(a, p, sz))

        data_lock.acquire()
        n_files[0] += 1
        if n_files[0] % 10 == 0:
            self.log().info("Read %d files (%.2f%%)" % (
                n_files[0], 100.0 * n_files[0] / total_files))
        data_lock.release()

    def load_data(self):
        if self.original_data != None and self.original_labels != None:
            return

        old_negative_data = []  # old negative set from previous snapshot
        old_file_map = []

        cached_data_fnme = "%s/%s_%s.pickle" % (
            config.cache_dir, os.path.basename(__file__),
                self.__class__.__name__)
        self.log().info("Will try to load previously cached data from "
                        "%s" % (cached_data_fnme))
        save_to_cache = True
        try:
            fin = open(cached_data_fnme, "rb")
            obj = pickle.load(fin)
            if obj["channels_dir"] != self.channels_dir:
                save_to_cache = False
                self.log().info("different dir found in cached data: %s" % (
                                                        obj["channels_dir"]))
                fin.close()
                raise FileNotFoundError()
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

            for k in self.pos.keys():
                self.log().info("%s: pos: (%.6f, %.6f)" % (
                    k, self.pos[k][0], self.pos[k][1]))
            self.log().info("sz: (%.6f, %.6f)" % (self.sz[0], self.sz[1]))
            self.log().info("rect: (%d, %d)" % (self.rect[0], self.rect[1]))

            self.shuffled_indexes = pickle.load(fin)
            self.original_labels = pickle.load(fin)
            sh = [self.original_labels.shape[0]]
            if not self.grayscale:
                sh.append(3)
            sh.append(self.rect[1])
            sh.append(self.rect[0])
            # Get raw array from file
            self.original_data = numpy.fromfile(fin, dtype=numpy.float32,
                                    count=numpy.prod(sh)).reshape(sh)
            self.rnd[0].state = pickle.load(fin)
            fin.close()
            self.log().info("Succeeded")
            self.log().info("class_samples=[%s]" % (
                ", ".join(str(x) for x in self.class_samples)))
            if self.w_neg == None or self.find_negative <= 0:
                return
            self.log().info("Will search for a negative set")
            # Saving the old negative set
            self.log().info("Saving the old negative set")
            for i in self.shuffled_indexes:
                l = self.original_labels[i]
                if l:
                    continue
                old_negative_data.append(self.original_data[i].copy())
                old_file_map.append(self.file_map[i])
            self.original_data = None
            self.original_labels = None
            self.shuffled_indexes = None
            self.log().info("Done")
        except FileNotFoundError:
            self.log().info("Failed")

        self.log().info("Will load data from original jp2 files")

        self.file_map.clear()

        # Read top-level configuration
        try:
            fin = open("%s/conf.py" % (self.channels_dir), "r")
            s = fin.read()
            fin.close()
            self.top_conf_ = {}
            exec(s, self.top_conf_, self.top_conf_)
        except:
            self.log().error("Error while executing %s/conf.py" % (
                self.channels_dir))
            raise

        # Read subdirectories configurations
        self.subdir_conf_.clear()
        for subdir in self.top_conf_["dirs_to_scan"]:
            try:
                fin = open("%s/%s/conf.py" % (self.channels_dir, subdir), "r")
                s = fin.read()
                fin.close()
                self.subdir_conf_[subdir] = {}
                exec(s, self.subdir_conf_[subdir], self.subdir_conf_[subdir])
            except:
                self.log().error("Error while executing %s/%s/conf.py" % (
                    self.channels_dir, subdir))
                raise

        # Parse configs
        self.channel_map = self.top_conf_["channel_map"]
        pos = {}
        rpos = {}
        for subdir, subdir_conf in self.subdir_conf_.items():
            frame = subdir_conf["frame"]
            if subdir not in pos.keys():
                pos[subdir] = frame.copy()  # bottom-right corner
                rpos[subdir] = [0, 0]
            for pos_size in subdir_conf["channel_map"].values():
                pos[subdir][0] = min(pos[subdir][0], pos_size["pos"][0])
                pos[subdir][1] = min(pos[subdir][1], pos_size["pos"][1])
                rpos[subdir][0] = max(rpos[subdir][0],
                                      pos_size["pos"][0] + pos_size["size"][0])
                rpos[subdir][1] = max(rpos[subdir][1],
                                      pos_size["pos"][1] + pos_size["size"][1])
            # Convert to relative values
            pos[subdir][0] /= frame[0]
            pos[subdir][1] /= frame[1]
            rpos[subdir][0] /= frame[0]
            rpos[subdir][1] /= frame[1]

        self.log().info("Found rectangles:")
        sz = [0, 0]
        for k in pos.keys():
            sz[0] = max(sz[0], rpos[k][0] - pos[k][0])
            sz[1] = max(sz[1], rpos[k][1] - pos[k][1])
            self.log().info("%s: pos: (%.6f, %.6f)" % (k, pos[k][0],
                                                       pos[k][1]))
        self.log().info("sz: (%.6f, %.6f)" % (sz[0], sz[1]))

        self.log().info("Adjusted rectangles:")
        sz[0] *= 1.02
        sz[1] *= 1.02
        self.log().info("sz: (%.6f, %.6f)" % (sz[0], sz[1]))
        for k in pos.keys():
            pos[k][0] += (rpos[k][0] - pos[k][0] - sz[0]) * 0.5
            pos[k][1] += (rpos[k][1] - pos[k][1] - sz[1]) * 0.5
            pos[k][0] = min(pos[k][0], 1.0 - sz[0])
            pos[k][1] = min(pos[k][1], 1.0 - sz[1])
            pos[k][0] = max(pos[k][0], 0.0)
            pos[k][1] = max(pos[k][1], 0.0)
            self.log().info("%s: pos: (%.6f, %.6f)" % (k, pos[k][0],
                                                       pos[k][1]))

        self.pos.clear()
        self.pos.update(pos)
        self.sz.clear()
        self.sz.extend(sz)

        max_lbl = 0
        files = {}
        total_files = 0
        baddir = re.compile("bad", re.IGNORECASE)
        jp2 = re.compile("\.jp2$", re.IGNORECASE)
        for subdir, subdir_conf in self.subdir_conf_.items():
            for dirnme in subdir_conf["channel_map"].keys():
                max_lbl = max(max_lbl, int(dirnme))
                relpath = "%s/%s" % (subdir, dirnme)
                found_files = []
                fordel = []
                for basedir, dirlist, filelist in os.walk("%s/%s" % (
                    self.channels_dir, relpath)):
                    for i, nme in enumerate(dirlist):
                        if baddir.search(nme) != None:
                            fordel.append(i)
                    while len(fordel) > 0:
                        dirlist.pop(fordel.pop())
                    for nme in filelist:
                        if jp2.search(nme) != None:
                            found_files.append("%s/%s" % (basedir, nme))
                found_files.sort()
                files[relpath] = found_files
                total_files += len(found_files)
        self.log().info("Found %d files" % (total_files))

        self.original_labels = numpy.zeros(
            total_files + len(old_negative_data),
            dtype=config.itypes[config.get_itype_from_size(max_lbl + 1)])
        if self.grayscale:
            self.original_data = numpy.zeros([
                total_files + len(old_negative_data),
                self.rect[1], self.rect[0]], numpy.float32)
        else:
            self.original_data = numpy.zeros([
                total_files + len(old_negative_data), 3,
                self.rect[1], self.rect[0]], numpy.float32)

        # Read samples in parallel
        rand = rnd.Rand()
        rand.seed(numpy.fromfile("/dev/urandom", dtype=numpy.int32,
                                 count=1024))
        n_threads = 32
        pool = thread_pool.ThreadPool(max_threads=n_threads,
                                      max_enqueued_tasks=n_threads)
        data_lock = threading.Lock()
        n_files = [0]
        i_sample = 0
        negative_data = {}  # dictionary: i => list of found negative data
        negative_file_map = {}  # dictionary: i => file name
        for subdir in sorted(self.subdir_conf_.keys()):
            subdir_conf = self.subdir_conf_[subdir]
            for dirnme in sorted(subdir_conf["channel_map"].keys()):
                relpath = "%s/%s" % (subdir, dirnme)
                self.log().info("Will load from %s" % (relpath))
                lbl = int(dirnme)
                for fnme in files[relpath]:
                    pool.request(self.from_jp2_async, ("%s" % (fnme),
                        pos[subdir], sz, data_lock,
                        0 + i_sample, 0 + lbl, n_files, total_files,
                        negative_data, negative_file_map, rand))
                    i_sample += 1
        pool.shutdown(execute_remaining=True)

        # Fill the negative data from previous pickle
        for i in range(len(old_negative_data)):
            self.original_data[total_files + i] = old_negative_data[i]
            self.file_map[total_files + i] = old_file_map[i]
        del(old_negative_data)
        del(old_file_map)

        # Check the need of negative data filtering
        if len(negative_data) > 0:
            n = 0
            for batch in negative_data.values():
                n += len(batch)
            self.log().info("Will filter %d negative samples" % (n))
            sh = [n]
            sh.extend(self.original_data[0].shape)
            data = numpy.empty(sh, dtype=self.original_data.dtype)
            file_map = {}
            j = 0
            for i in sorted(negative_data.keys()):
                for sample in negative_data[i]:
                    data[j] = sample
                    file_map[j] = negative_file_map[i]
                    j += 1
                negative_data[i].clear()
            del(negative_data)
            del(negative_file_map)
            lbls = numpy.zeros(n, dtype=numpy.int8)
            idxs = numpy.arange(n,
                dtype=config.itypes[config.get_itype_from_size(n)])
            n = self.filter_negative(data, lbls, idxs)
            self.w_neg.loader.original_data = None
            self.w_neg.loader.original_labels = None
            self.w_neg.loader.shuffled_indexes = None
            self.w_neg = None
            nn = len(self.original_data)
            # Saving extracted negative samples
            dirnme = "%s/found_negative_images" % (config.cache_dir)
            self.log().info("Dumping found negative images to %s" % (dirnme))
            try:
                os.mkdir(dirnme)
            except OSError:
                pass
            for i in range(n):
                fnme = "%s/%d.png" % (dirnme, nn + i)
                scipy.misc.imsave(fnme, self.as_image(data[i]))
            self.log().info("Done")
            #
            self.original_data = numpy.append(self.original_data, data[:n],
                                              axis=0)
            self.original_labels = numpy.append(self.original_labels, lbls[:n],
                                                axis=0)
            for i in range(n):
                self.file_map[nn + i] = file_map[idxs[i]]
            del(file_map)

        self.class_samples[0] = 0
        self.class_samples[1] = 0
        self.class_samples[2] = self.original_data.shape[0]

        # Randomly generate validation set from train.
        self.extract_validation_from_train()

        # Saving all the samples
        self.log().info("Dumping all the samples to %s" % (config.cache_dir))
        for i in self.shuffled_indexes:
            l = self.original_labels[i]
            dirnme = "%s/%03d" % (config.cache_dir, l)
            try:
                os.mkdir(dirnme)
            except OSError:
                pass
            fnme = "%s/%d.png" % (dirnme, i)
            scipy.misc.imsave(fnme, self.as_image(self.original_data[i]))
        self.log().info("Done")
        #

        self.log().info("class_samples=[%s]" % (
            ", ".join(str(x) for x in self.class_samples)))

        if not save_to_cache:
            return
        self.log().info("Saving loaded data for later faster load to "
                        "%s" % (cached_data_fnme))
        fout = open(cached_data_fnme, "wb")
        obj = {}
        for name in self.attributes_for_cached_data:
            obj[name] = self.__dict__[name]
        pickle.dump(obj, fout)
        pickle.dump(self.shuffled_indexes, fout)
        pickle.dump(self.original_labels, fout)
        # Because pickle doesn't support greater than 4Gb arrays
        self.original_data.ravel().tofile(fout)
        # Save random state
        pickle.dump(self.rnd[0].state, fout)
        fout.close()
        self.log().info("Done")

    def filter_negative(self, data, lbls, idxs):
        """Filters negative data by running w_neg workflow over it.
        """
        w_neg = self.w_neg
        w_neg.loader.original_data = data
        w_neg.loader.original_labels = lbls
        w_neg.loader.shuffled_indexes = idxs
        w_neg.loader.class_samples[0] = 0
        w_neg.loader.class_samples[1] = 0
        w_neg.loader.class_samples[2] = len(data)

        w_neg.decision.unlink()
        w_neg.decision.link_from(w_neg.ev)
        for f in w_neg.forward:
            f.device = w_neg.device
        w_neg.ev.device = w_neg.device
        w_neg.saver = Saver()
        w_neg.saver.vectors_to_save["m"] = w_neg.forward[-1].max_idx
        w_neg.saver.link_from(w_neg.decision)
        w_neg.loader.shuffle = w_neg.loader.nothing
        w_neg.loader.shuffle_train = w_neg.loader.nothing
        w_neg.loader.shuffle_validation_train = w_neg.loader.nothing
        w_neg.decision.on_snapshot = w_neg.decision.nothing
        w_neg.saver.minibatch_size = w_neg.loader.minibatch_size
        w_neg.end_point.link_from(w_neg.saver)
        w_neg.rpt.link_from(w_neg.saver)
        w_neg.loader.gate_block = w_neg.decision.epoch_ended
        w_neg.end_point.gate_block = w_neg.decision.epoch_ended
        w_neg.end_point.gate_block_not = [1]
        w_neg.end_point.link_from(w_neg.saver)
        for gd in w_neg.gd:
            gd.unlink()
        w_neg.start_point.initialize_dependent()
        w_neg.run()

        n = 0
        i = 0
        for batch in w_neg.saver.vectors_["m"]:
            for j in batch:
                if j != lbls[i]:
                    data[n] = data[i]
                    lbls[n] = lbls[i]
                    idxs[n] = idxs[i]
                    n += 1
                i += 1
        return n

    def as_image(self, x):
        if len(x.shape) == 2:
            x = x.copy()
        elif len(x.shape) == 3:
            if x.shape[2] == 3:
                x = x.copy()
            elif x.shape[0] == 3:
                xx = numpy.empty([x.shape[1], x.shape[2], 3],
                                 dtype=x.dtype)
                xx[:, :, 0:1] = x[0:1, :, :].reshape(
                    x.shape[1], x.shape[2], 1)[:, :, 0:1]
                xx[:, :, 1:2] = x[1:2, :, :].reshape(
                    x.shape[1], x.shape[2], 1)[:, :, 0:1]
                xx[:, :, 2:3] = x[2:3, :, :].reshape(
                    x.shape[1], x.shape[2], 1)[:, :, 0:1]
                x = xx
            else:
                raise error.ErrBadFormat()
        else:
            raise error.ErrBadFormat()
        return formats.norm_image(x, True)


class Workflow(workflow.NNWorkflow):
    """Workflow.
    """
    def __init__(self, layers=None, device=None):
        super(Workflow, self).__init__(device=device)

        self.saver = None

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
            "/tmp/tmpimg/test",
            "/tmp/tmpimg/validation",
            "/tmp/tmpimg/train"], yuv=True)
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
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        self.image_saver.gate_skip = self.decision.just_snapshotted
        self.image_saver.gate_skip_not = [1]
        self.image_saver.this_save_time = self.decision.snapshot_time

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(device=device)
        # self.gd[-1].link_from(self.decision)
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

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotters.SimplePlotter(figure_label="num errors",
                                                   plot_style=styles[i],
                                                   ylim=(0, 100)))
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision if len(self.plt) == 1
                                   else self.plt[-2])
            self.plt[-1].gate_skip = self.decision.epoch_ended
            self.plt[-1].gate_skip_not = [1]
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Weights plotter
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_w = plotters.Weights2D(figure_label="First Layer Weights",
                                        limit=16, yuv=True)
        self.plt_w.input = self.gd[0].weights
        self.plt_w.get_shape_from = self.forward[0].input
        self.plt_w.input_field = "v"
        self.plt_w.link_from(self.plt[-1])
        self.plt_w.gate_skip = self.decision.epoch_ended
        self.plt_w.gate_skip_not = [1]
        # Image plottter
        self.decision.vectors_to_sync[self.forward[0].input] = 1
        self.decision.vectors_to_sync[self.ev.labels] = 1
        self.plt_i = plotters.Image(figure_label="Input", yuv=True)
        self.plt_i.inputs.append(self.decision)
        self.plt_i.input_fields.append("sample_label")
        self.plt_i.inputs.append(self.decision)
        self.plt_i.input_fields.append("sample_input")
        self.plt_i.link_from(self.plt_w)
        self.plt_i.gate_skip = self.decision.epoch_ended
        self.plt_i.gate_skip_not = [1]
        # Confusion matrix plotter
        self.plt_mx = []
        j = 0
        for i in range(1, 3):
            self.plt_mx.append(plotters.MatrixPlotter(
                figure_label=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.plt_mx[-2] if j else self.plt_i)
            self.plt_mx[-1].gate_skip = self.decision.epoch_ended
            self.plt_mx[-1].gate_skip_not = [1]
            j += 1
        self.gd[-1].link_from(self.plt_mx[-1])

    def initialize(self, global_alpha, global_lambda,
                   minibatch_maxsize, dirnme, dump,
                   snapshot_prefix, w_neg, find_negative, device):
        self.decision.snapshot_prefix = snapshot_prefix
        self.loader.channels_dir = dirnme
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        self.loader.w_neg = w_neg
        self.loader.find_negative = find_negative
        self.ev.device = device
        for gd in self.gd:
            gd.device = device
            gd.global_alpha = global_alpha
            gd.global_lambda = global_lambda
        for forward in self.forward:
            forward.device = device
        if self.__dict__.get("saver") != None:
            self.saver.unlink()
            self.saver = None
        if len(dump):
            self.saver = Saver(fnme=dump)
            self.saver.link_from(self.decision)
            self.loader.shuffle = self.loader.nothing
            self.loader.shuffle_train = self.loader.nothing
            self.loader.shuffle_validation_train = self.loader.nothing
            #self.forward[-1].gpu_apply_exp = self.forward[-1].nothing
            #self.forward[-1].cpu_apply_exp = self.forward[-1].nothing
            self.decision.on_snapshot = self.decision.nothing
            self.saver.vectors_to_save["y"] = self.forward[-1].output
            self.saver.vectors_to_save["l"] = self.loader.minibatch_labels
            self.saver.flush = self.decision.epoch_ended
            self.saver.minibatch_size = self.loader.minibatch_size
            self.end_point.link_from(self.saver)
            self.rpt.link_from(self.saver)
            self.loader.gate_block = self.decision.epoch_ended
            self.end_point.gate_block = self.decision.epoch_ended
            self.end_point.gate_block_not = [1]
            self.end_point.link_from(self.plt_mx[-1])
            for gd in self.gd:
                gd.unlink()
        return self.start_point.initialize_recursively()


class Saver(units.Unit):
    """Saves vars to file.

    Attributes:
        vectors_to_save: dictionary of vectors to save,
            name => Vector().
        vectors_: dictionary of lists of accumulated vectors.
        flush: if [1] - flushes vectors_ to fnme.
        fnme: filename to save vectors_ to.
    """
    def __init__(self, fnme=None):
        super(Saver, self).__init__()
        self.vectors_to_save = {}
        self.vectors_ = {}
        self.flush = [0]
        self.fnme = fnme
        self.minibatch_size = None

    def run(self):
        for name, vector in self.vectors_to_save.items():
            vector.sync()
            if name not in self.vectors_.keys():
                self.vectors_[name] = []
            if self.minibatch_size != None:
                self.vectors_[name].append(
                    vector.v[:self.minibatch_size[0]].copy())
            else:
                self.vectors_[name].append(vector.v.copy())
        if self.flush[0] and self.fnme != None:
            self.log().info("Saving collected vectors to %s" % (self.fnme))
            to_save = {}
            for name, vectors in self.vectors_.items():
                sh = [0]
                dtype = config.dtypes[config.dtype]
                for vector in vectors:
                    if len(sh) == 1:
                        dtype = vector.dtype
                        sh.extend(vector.shape[1:])
                    sh[0] += len(vector)
                a = numpy.zeros(sh, dtype=dtype)
                i = 0
                for vector in vectors:
                    n = len(vector)
                    a[i:i + n] = vector
                    i += n
                to_save[name] = a
                self.vectors_[name].clear()
            self.vectors_.clear()
            scipy.io.savemat(self.fnme, to_save, format='5',
                             long_field_names=True, oned_as='row')
            self.log().info("Saved")
            to_save.clear()
            time.sleep(86400)


import time
import traceback
import argparse


def main():
    """Some visualization
    import matplotlib.pyplot as pp
    cached_data_fnme = "%s/%s_Loader.pickle" % (
        config.cache_dir, os.path.basename(__file__))
    fin = open(cached_data_fnme, "rb")
    pickle.load(fin)
    original_labels = pickle.load(fin)
    a = pickle.load(fin)
    sh = [original_labels.shape[0]]
    sh.extend(a.shape)
    original_data = numpy.zeros(sh, dtype=numpy.float32)
    original_data[0] = a
    for i in range(1, original_data.shape[0]):
        a = pickle.load(fin)
        original_data[i] = a
    fin.close()
    fig = pp.figure()
    for i in range(49):
        ax = fig.add_subplot(7, 7, i + 1)
        ax.axis('off')

        ii = numpy.random.randint(original_data.shape[0])
        a = original_data[ii]

        aa = numpy.zeros([a.shape[1], a.shape[2], 3], dtype=numpy.float32)
        aa[:, :, 0:1] = a[0:1, :, :].reshape(a.shape[1], a.shape[2], 1)[:, :,
                                                                        0:1]
        aa[:, :, 1:2] = a[1:2, :, :].reshape(a.shape[1], a.shape[2], 1)[:, :,
                                                                        0:1]
        aa[:, :, 2:3] = a[2:3, :, :].reshape(a.shape[1], a.shape[2], 1)[:, :,
                                                                        0:1]
        aa -= aa.min()
        m = aa.max()
        if m:
            aa /= m
            aa *= 255.0

        ax.imshow(aa.astype(numpy.uint8), interpolation="nearest")
        ax.text(0.5, 0.5, str(original_labels[ii]),
                horizontalalignment='center',
                verticalalignment='center')
    pp.show()
    sys.exit(0)
    """
    #if __debug__:
    #    logging.basicConfig(level=logging.DEBUG)
    #else:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str,
        help="Snapshot with trained network",
        default="%s/channels_kor.pickle" % (config.snapshot_dir))
    parser.add_argument("-export", type=bool,
        help="Export trained network to C",
        default=False)
    parser.add_argument("-dump", type=str,
        help="Dump trained network output to .mat",
        default="")
    parser.add_argument("-dir", type=str,
        help="Directory with channels",
        default="%s/channels/korean_960_540/train" % (
                                    config.test_dataset_root))
    parser.add_argument("-snapshot_prefix", type=str,
        help="Snapshot prefix.", default="108_24")
    parser.add_argument("-layers", type=str,
        help="NN layer sizes, separated by any separator.",
        default="108_24")
    parser.add_argument("-minibatch_size", type=int,
        help="Minibatch size.", default=81)
    parser.add_argument("-global_alpha", type=float,
        help="Global Alpha.", default=0.01)
    parser.add_argument("-global_lambda", type=float,
        help="Global Lambda.", default=0.00005)
    parser.add_argument("-find_negative", type=int,
        help="Extend negative dataset by at most this number of negative "
             "samples per image. -snapshot should be provided.",
        default=0)
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
    # rnd.default.seed(numpy.fromfile("/dev/urandom", numpy.int32, 1024))
    cl = opencl.DeviceList()
    device = cl.get_device()
    w_neg = None
    try:
        fin = open(args.snapshot, "rb")
        w = pickle.load(fin)
        fin.close()
        if args.export:
            tm = time.localtime()
            s = "%d.%02d.%02d_%02d.%02d.%02d" % (
                tm.tm_year, tm.tm_mon, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec)
            fnme = "%s/kor_channels_workflow_%s" % (config.snapshot_dir, s)
            try:
                w.export(fnme)
                logging.info("Exported successfully to %s" % (fnme))
            except:
                a, b, c = sys.exc_info()
                traceback.print_exception(a, b, c)
                logging.error("Error while exporting.")
            sys.exit(0)
        if args.find_negative > 0:
            w_neg = w
            w_neg.device = device
            raise IOError()
    except IOError:
        w = Workflow(layers=layers, device=device)
    w.initialize(global_alpha=args.global_alpha,
                 global_lambda=args.global_lambda,
                 minibatch_maxsize=args.minibatch_size, dirnme=args.dir,
                 dump=args.dump, snapshot_prefix=args.snapshot_prefix,
                 w_neg=w_neg, find_negative=args.find_negative, device=device)
    fnme = "%s/%s.txt" % (config.cache_dir, args.snapshot_prefix)
    logging.info("Dumping file map to %s" % (fnme))
    fout = open(fnme, "w")
    file_map = w.loader.file_map
    for i in sorted(file_map.keys()):
        fout.write("%d\t%s\n" % (i, file_map[i]))
    fout.close()
    logging.info("Done")
    logging.info("Will execute workflow now")
    w.run()
    plotters.Graphics().wait_finish()
    logging.info("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
