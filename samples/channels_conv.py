#!/usr/bin/python3.3 -O

"""
Created on April 22, 2014

Convolitional channels recognition.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import glymur
import logging
import numpy
import os
import pickle
import re
import scipy.misc
import sys
import threading
import time
import traceback

# FIXME(a.kazantsev): numpy.dot works 5 times faster with this option
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from veles.config import root
import veles.error as error
import veles.formats as formats
import veles.image as image
from veles.mutable import Bool
import veles.plotting_units as plotting_units
import veles.rnd as rnd
import veles.thread_pool as thread_pool
import veles.znicz.accumulator as accumulator
import veles.znicz.all2all as all2all
import veles.znicz.conv as conv
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.gd_conv as gd_conv
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.image_saver as image_saver
import veles.znicz.loader as loader
import veles.znicz.nn_units as nn_units
import veles.znicz.pooling as pooling

if (sys.version_info[0] + (sys.version_info[1] / 10.0)) < 3.3:
    FileNotFoundError = IOError  # pylint: disable=W0622

root.defaults = {"accumulator": {"n_bars": 30},
                 "decision": {"fail_iterations": 1000,
                              "snapshot_prefix": "channels_conv_108_24",
                              "use_dynamic_alpha": False,
                              "do_export_weights": True},
                 "image_saver": {"out_dirs":
                                 [os.path.join(root.common.cache_dir,
                                               "tmp/test"),
                                  os.path.join(root.common.cache_dir,
                                               "tmp/validation"),
                                  os.path.join(root.common.cache_dir,
                                               "tmp/train")]},
                 "loader": {"cache_fnme": os.path.join(root.common.cache_dir,
                                                       "channels_conv.pickle"),
                            "grayscale": False,
                            "minibatch_size": 81,
                            "n_threads": 32,
                            "channels_dir":
                            "/data/veles/TV/russian_small/train",
                            "rect": (264, 129),
                            "validation_procent": 0.15},
                 "weights_plotter": {"limit": 64},
                 "channels_conv": {"export": False,
                                   "find_negative": 0,
                                   "global_alpha": 0.001,
                                   "global_lambda": 0.004,
                                   "layers":
                                   [{"type": "conv", "n_kernels": 32,
                                     "kx": 5, "ky": 5,
                                     "padding": (2, 2, 2, 2)},
                                    {"type": "max_pooling",
                                     "kx": 3, "ky": 3, "sliding": (2, 2)},
                                    {"type": "conv", "n_kernels": 32,
                                     "kx": 5, "ky": 5, "padding":
                                     (2, 2, 2, 2)},
                                    {"type": "avg_pooling",
                                     "kx": 3, "ky": 3, "sliding": (2, 2)},
                                    {"type": "conv", "n_kernels": 64,
                                     "kx": 5, "ky": 5,
                                     "padding": (2, 2, 2, 2)},
                                    {"type": "avg_pooling",
                                     "kx": 3, "ky": 3, "sliding": (2, 2)},
                                    {"type": "tanh", "layers": 10}],
                                   "snapshot": ""}}


class Loader(loader.FullBatchLoader):
    """Loads channels.
    """
    def __init__(self, workflow, **kwargs):
        channels_dir = kwargs.get("channels_dir", "")
        rect = kwargs.get("rect", (264, 129))
        grayscale = kwargs.get("grayscale", False)
        cache_fnme = kwargs.get("cache_fnme", "")
        find_negative = kwargs.get("find_negative", 0)
        n_threads = kwargs.get("n_threads", 32)
        kwargs["channels_dir"] = channels_dir
        kwargs["rect"] = rect
        kwargs["grayscale"] = grayscale
        kwargs["cache_fnme"] = cache_fnme
        kwargs["find_negative"] = find_negative
        kwargs["n_threads"] = n_threads
        super(Loader, self).__init__(workflow, **kwargs)
        # : Top-level configuration from channels_dir/conf.py
        self.top_conf_ = None
        # : Configuration from channels_dir/subdirectory/conf.py
        self.subdir_conf_ = {}
        self.channels_dir = channels_dir
        self.cache_fnme = cache_fnme
        self.rect = rect
        self.grayscale = grayscale
        self.w_neg = None  # workflow for finding the negative dataset
        self.find_negative = find_negative
        self.channel_map = None
        self.n_threads = n_threads
        self.pos = {}
        self.sz = {}
        self.file_map = {}  # sample index to its file name map
        self.attributes_for_cached_data = [
            "channels_dir", "rect", "channel_map", "pos", "sz",
            "class_samples", "grayscale", "file_map", "cache_fnme"]
        self.exports = ["rect", "pos", "sz"]

    def from_jp2(self, fnme):
        try:
            j2 = glymur.Jp2k(fnme)
        except:
            self.error("glymur.Jp2k() failed for %s" % (fnme))
            raise
        a2 = j2.read()
        if j2.box[2].box[1].colorspace == 16:  # RGB
            if self.grayscale:
                # Get Y component from RGB
                a = numpy.empty([a2.shape[0], a2.shape[1], 1],
                                dtype=numpy.uint8)
                a[:, :, 0:1] = numpy.clip(
                    0.299 * a2[:, :, 0:1] +
                    0.587 * a2[:, :, 1:2] +
                    0.114 * a2[:, :, 2:3], 0, 255)
                a = formats.reshape(a, [a2.shape[0], a2.shape[1]])
            else:
                # Convert to YUV
                # Y = 0.299 * R + 0.587 * G + 0.114 * B;
                # U = -0.14713 * R - 0.28886 * G + 0.436 * B + 128;
                # V = 0.615 * R - 0.51499 * G - 0.10001 * B + 128;
                # and transform to different planes
                a = numpy.empty([3, a2.shape[0], a2.shape[1]],
                                dtype=numpy.uint8)
                a[0:1, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = numpy.clip(
                    0.299 * a2[:, :, 0:1] +
                    0.587 * a2[:, :, 1:2] +
                    0.114 * a2[:, :, 2:3], 0, 255)
                a[1:2, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = numpy.clip(
                    (-0.14713) * a2[:, :, 0:1] +
                    (-0.28886) * a2[:, :, 1:2] +
                    0.436 * a2[:, :, 2:3] + 128, 0, 255)
                a[2:3, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = numpy.clip(
                    0.615 * a2[:, :, 0:1] +
                    (-0.51499) * a2[:, :, 1:2] +
                    (-0.10001) * a2[:, :, 2:3] + 128, 0, 255)
        elif j2.box[2].box[1].colorspace == 18:  # YUV
            if self.grayscale:
                a = numpy.empty([a2.shape[0], a2.shape[1], 1],
                                dtype=numpy.uint8)
                a[:, :, 0:1] = a2[:, :, 0:1]
                a = formats.reshape(a, [a2.shape[0], a2.shape[1]])
            else:
                # transform to different yuv planes
                a = numpy.empty([3, a2.shape[0], a2.shape[1]],
                                dtype=numpy.uint8)
                a[0:1, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 0:1]
                a[1:2, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 1:2]
                a[2:3, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 2:3]
        else:
            raise error.ErrBadFormat("Unknown colorspace in %s" % (fnme))
        return a

    def sample_rect(self, a, pos, sz):
        if self.grayscale:
            aa = numpy.empty([self.rect[1], self.rect[0]], dtype=numpy.float32)
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
            aa = numpy.empty([3, self.rect[1], self.rect[0]],
                             dtype=numpy.float32)
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

    def append_sample(self, sample, lbl, fnme, n_negative, data_lock):
        data_lock.acquire()
        self.original_data.append(sample)
        self.original_labels.append(lbl)
        ii = len(self.original_data) - 1
        self.file_map[ii] = fnme
        if n_negative is not None:
            n_negative[0] += 1
        data_lock.release()
        return ii

    def from_jp2_async(self, fnme, pos, sz, data_lock, stat_lock,
                       i_sample, lbl, n_files, total_files,
                       n_negative, rand):
        """Loads, crops and normalizes image in the parallel thread.
        """
        a = self.from_jp2(fnme)

        sample = self.sample_rect(a, pos, sz)
        sample = numpy.swapaxes(sample, 0, 1)
        sample = numpy.swapaxes(sample, 1, 2)

        self.append_sample(sample, lbl, fnme, None, data_lock)

        # Collect negative dataset from positive samples only
        if lbl and self.w_neg is not None and self.find_negative > 0:
            # Sample pictures at random positions
            samples = numpy.zeros([self.find_negative, sample.size],
                                  dtype=self.w_neg[0][0].dtype)
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
                samples[i][:] = self.sample_rect(a, p, sz).ravel()[:]
            ll = self.get_labels_from_samples(samples)
            for i, l in enumerate(ll):
                if l == 0:
                    continue
                # negative found
                s = samples[i].reshape(sample.shape)
                ii = self.append_sample(s, 0, fnme, n_negative, data_lock)
                dirnme = "%s/found_negative_images" % (root.common.cache_dir)
                try:
                    os.mkdir(dirnme)
                except OSError:
                    pass
                fnme = "%s/0_as_%d.%d.png" % (dirnme, l, ii)
                scipy.misc.imsave(fnme, self.as_image(s))

        stat_lock.acquire()
        n_files[0] += 1
        if n_files[0] % 10 == 0:
            self.info("Read %d files (%.2f%%)" % (
                n_files[0], 100.0 * n_files[0] / total_files))
        stat_lock.release()

    def get_labels_from_samples(self, samples):
        weights = self.w_neg[0]
        bias = self.w_neg[1]
        n = len(weights)
        a = samples
        for i in range(n):
            a = numpy.dot(a, weights[i].transpose())
            a += bias[i]
            if i < n - 1:
                a *= 0.6666
                numpy.tanh(a, a)
                a *= 1.7159
        return a.argmax(axis=1)

    def get_label(self, dirnme):
        lbl = self.channel_map[dirnme].get("lbl")
        if lbl is None:
            lbl = int(dirnme)
        return lbl

    def load_data(self):
        if self.original_data is not None and self.original_labels is not None:
            return

        cached_data_fnme = (
            os.path.join(
                root.common.cache_dir,
                "%s_%s.pickle" %
                (os.path.basename(__file__), self.__class__.__name__))
            if not len(self.cache_fnme) else self.cache_fnme)
        self.info("Will try to load previously cached data from " +
                  cached_data_fnme)
        save_to_cache = True
        try:
            fin = open(cached_data_fnme, "rb")
            obj = pickle.load(fin)
            if obj["channels_dir"] != self.channels_dir:
                save_to_cache = False
                self.info("different dir found in cached data: %s" % (
                    obj["channels_dir"]))
                fin.close()
                raise FileNotFoundError()
            for k, v in obj.items():
                if type(v) == list:
                    o = self.__dict__[k]
                    if o is None:
                        o = []
                        self.__dict__[k] = o
                    del o[:]
                    o.extend(v)
                elif type(v) == dict:
                    o = self.__dict__[k]
                    if o is None:
                        o = {}
                        self.__dict__[k] = o
                    o.update(v)
                else:
                    self.__dict__[k] = v

            for k in self.pos.keys():
                self.info("%s: pos=(%.6f, %.6f) sz=(%.6f, %.6f)" % (
                    k, self.pos[k][0], self.pos[k][1],
                    self.sz[k][0], self.sz[k][1]))
            self.info("rect: (%d, %d)" % (self.rect[0], self.rect[1]))

            self.shuffled_indexes = pickle.load(fin)
            self.original_labels = pickle.load(fin)
            sh = ([self.rect[1], self.rect[0]] if self.grayscale
                  else [self.rect[1], self.rect[0], 3])
            n = int(numpy.prod(sh))
            # Get raw array from file
            self.original_data = []
            store_negative = self.w_neg is not None and self.find_negative > 0
            old_file_map = []
            n_not_exists_anymore = 0
            for i in range(len(self.original_labels)):
                a = numpy.fromfile(fin, dtype=numpy.float32, count=n)
                if store_negative:
                    if self.original_labels[i]:
                        del a
                        continue
                    if not os.path.isfile(self.file_map[i]):
                        n_not_exists_anymore += 1
                        del a
                        continue
                    old_file_map.append(self.file_map[i])
                self.original_data.append(a.reshape(sh))
            self.rnd[0].state = pickle.load(fin)
            fin.close()
            self.info("Succeeded")
            self.info("class_samples=[%s]" % (
                ", ".join(str(x) for x in self.class_samples)))
            if not store_negative:
                return
            self.info("Will search for a negative set at most %d "
                      "samples per image" % (self.find_negative))
            # Saving the old negative set
            self.info("Extracting the old negative set")
            self.file_map.clear()
            for i, fnme in enumerate(old_file_map):
                self.file_map[i] = fnme
            del old_file_map
            n = len(self.original_data)
            self.original_labels = list(0 for i in range(n))
            self.shuffled_indexes = None
            self.info("Done (%d extracted, %d not exists anymore)" % (
                n, n_not_exists_anymore))
        except FileNotFoundError:
            self.info("Failed")
            self.original_labels = []
            self.original_data = []
            self.shuffled_indexes = None
            self.file_map.clear()

        self.info("Will load data from original jp2 files")

        # Read top-level configuration
        try:
            fin = open(os.path.join(self.channels_dir, "conf.py"), "r")
            s = fin.read()
            fin.close()
            self.top_conf_ = {}
            exec(s, self.top_conf_, self.top_conf_)
        except:
            self.error("Error while executing %s/conf.py" % (
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
                self.error("Error while executing %s/%s/conf.py" % (
                    self.channels_dir, subdir))
                raise

        # Parse configs
        self.channel_map = self.top_conf_["channel_map"]
        pos = {}
        rpos = {}
        sz = {}
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
            sz[subdir] = [rpos[subdir][0] - pos[subdir][0],
                          rpos[subdir][1] - pos[subdir][1]]

        self.info("Found rectangles:")
        for k in pos.keys():
            self.info("%s: pos=(%.6f, %.6f) sz=(%.6f, %.6f)" % (
                k, pos[k][0], pos[k][1], sz[k][0], sz[k][1]))

        self.info("Adjusted rectangles:")
        for k in pos.keys():
            # sz[k][0] *= 1.01
            # sz[k][1] *= 1.01
            pos[k][0] += (rpos[k][0] - pos[k][0] - sz[k][0]) * 0.5
            pos[k][1] += (rpos[k][1] - pos[k][1] - sz[k][1]) * 0.5
            pos[k][0] = min(pos[k][0], 1.0 - sz[k][0])
            pos[k][1] = min(pos[k][1], 1.0 - sz[k][1])
            pos[k][0] = max(pos[k][0], 0.0)
            pos[k][1] = max(pos[k][1], 0.0)
            self.info("%s: pos=(%.6f, %.6f) sz=(%.6f, %.6f)" % (
                k, pos[k][0], pos[k][1], sz[k][0], sz[k][1]))

        self.pos.clear()
        self.pos.update(pos)
        self.sz.clear()
        self.sz.update(sz)

        max_lbl = 0
        files = {}
        total_files = 0
        baddir = re.compile("bad", re.IGNORECASE)
        jp2 = re.compile("\.jp2$", re.IGNORECASE)
        for subdir, subdir_conf in self.subdir_conf_.items():
            for dirnme in subdir_conf["channel_map"].keys():
                max_lbl = max(max_lbl, self.get_label(dirnme))
                relpath = "%s/%s" % (subdir, dirnme)
                found_files = []
                fordel = []
                for basedir, dirlist, filelist in os.walk(
                        "%s/%s" % (self.channels_dir, relpath)):
                    for i, nme in enumerate(dirlist):
                        if baddir.search(nme) is not None:
                            fordel.append(i)
                    while len(fordel) > 0:
                        dirlist.pop(fordel.pop())
                    for nme in filelist:
                        if jp2.search(nme) is not None:
                            found_files.append("%s/%s" % (basedir, nme))
                found_files.sort()
                files[relpath] = found_files
                total_files += len(found_files)
        self.info("Found %d files" % (total_files))

        # Read samples in parallel
        rand = rnd.Rand()
        rand.seed(numpy.fromfile("/dev/urandom", dtype=numpy.int32,
                                 count=1024))
        # FIXME(a.kazantsev): numpy.dot is thread-safe with this value
        # on ubuntu 13.10 (due to the static number of buffers in libopenblas)
        n_threads = self.n_threads
        pool = thread_pool.ThreadPool(minthreads=1, maxthreads=n_threads,
                                      queue_size=n_threads)
        data_lock = threading.Lock()
        stat_lock = threading.Lock()
        n_files = [0]
        n_negative = [0]
        i_sample = 0
        for subdir in sorted(self.subdir_conf_.keys()):
            subdir_conf = self.subdir_conf_[subdir]
            for dirnme in sorted(subdir_conf["channel_map"].keys()):
                relpath = "%s/%s" % (subdir, dirnme)
                self.info("Will load from %s" % (relpath))
                lbl = self.get_label(dirnme)
                for fnme in files[relpath]:
                    pool.request(self.from_jp2_async, (
                        fnme, pos[subdir], sz[subdir],
                        data_lock, stat_lock,
                        0 + i_sample, 0 + lbl, n_files, total_files,
                        n_negative, rand))
                    i_sample += 1
        pool.shutdown(execute_remaining=True)

        if (len(self.original_data) != len(self.original_labels) or
                len(self.file_map) != len(self.original_labels)):
            raise Exception("Logic error")

        if self.w_neg is not None and self.find_negative > 0:
            n_positive = numpy.count_nonzero(self.original_labels)
            self.info("Found %d negative samples (%.2f%%)" % (
                n_negative[0], 100.0 * n_negative[0] / n_positive))

        self.info("Loaded %d samples with resize and %d without" % (
            image.resize_count, image.asitis_count))

        self.class_samples[0] = 0
        self.class_samples[1] = 0
        self.class_samples[2] = len(self.original_data)

        # Randomly generate validation set from train.
        self.info("Will extract validation set from train")
        self.extract_validation_from_train(rnd.default2)

        # Saving all the samples
        """
        self.info("Dumping all the samples to %s" % (root.common.cache_dir))
        for i in self.shuffled_indexes:
            l = self.original_labels[i]
            dirnme = "%s/%03d" % (root.common.cache_dir, l)
            try:
                os.mkdir(dirnme)
            except OSError:
                pass
            fnme = "%s/%d.png" % (dirnme, i)
            scipy.misc.imsave(fnme, self.as_image(self.original_data[i]))
        self.info("Done")
        """

        self.info("class_samples=[%s]" % (
            ", ".join(str(x) for x in self.class_samples)))

        if not save_to_cache:
            return
        self.info("Saving loaded data for later faster load to "
                  "%s" % (cached_data_fnme))
        fout = open(cached_data_fnme, "wb")
        obj = {}
        for name in self.attributes_for_cached_data:
            obj[name] = self.__dict__[name]
        pickle.dump(obj, fout)
        pickle.dump(self.shuffled_indexes, fout)
        pickle.dump(self.original_labels, fout)
        # Because pickle doesn't support greater than 4Gb arrays
        for i in range(len(self.original_data)):
            self.original_data[i].ravel().tofile(fout)
        # Save random state
        pickle.dump(self.rnd[0].state, fout)
        fout.close()
        self.info("Done")

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


class Workflow(nn_units.NNWorkflow):
    """Workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.saver = None

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, cache_fnme=root.loader.cache_fnme,
                             find_negative=root.channels_conv.find_negative,
                             grayscale=root.loader.grayscale,
                             n_threads=root.loader.n_threads,
                             channels_dir=root.loader.channels_dir,
                             rect=root.loader.rect,
                             validation_procent=root.loader.validation_procent)
        self.loader.link_from(self.repeater)

        # Add forward units
        del self.forward[:]
        for i in range(0, len(layers)):
            layer = layers[i]
            if layer["type"] == "relu":
                if i == len(layers) - 1:
                    aa = all2all.All2AllSoftmax(
                        self, output_shape=[layer["layers"]], device=device)
                else:
                    aa = all2all.All2AllRELU(
                        self, output_shape=[layer["layers"]], device=device)
            elif layer["type"] == "tanh":
                print("layers", layer["layers"])
                if i == len(layers) - 1:
                    aa = all2all.All2AllSoftmax(
                        self, output_shape=[layer["layers"]], device=device)
                else:
                    aa = all2all.All2AllTanh(
                        self, output_shape=[layer["layers"]], device=device)
            elif layer["type"] == "conv":
                aa = conv.ConvTanh(self, n_kernels=layer["n_kernels"],
                                   kx=layer["kx"], ky=layer["ky"],
                                   sliding=layer.get("sliding", (1, 1, 1, 1)),
                                   padding=layer.get("padding", (0, 0, 0, 0)),
                                   device=device)
            elif layer["type"] == "conv_relu":
                aa = conv.ConvRELU(self, n_kernels=layer["n_kernels"],
                                   kx=layer["kx"], ky=layer["ky"],
                                   sliding=layer.get("sliding", (1, 1, 1, 1)),
                                   padding=layer.get("padding", (0, 0, 0, 0)),
                                   device=device,
                                   weights_filling="normal")
            elif layer["type"] == "max_pooling":
                aa = pooling.MaxPooling(self, kx=layer["kx"], ky=layer["ky"],
                                        sliding=layer.get("sliding",
                                                          (layer["kx"],
                                                           layer["ky"])),
                                        device=device)
            elif layer["type"] == "avg_pooling":
                aa = pooling.AvgPooling(self, kx=layer["kx"], ky=layer["ky"],
                                        sliding=layer.get("sliding",
                                                          (layer["kx"],
                                                           layer["ky"])),
                                        device=device)
            else:
                raise error.ErrBadFormat("Unsupported layer type %s" %
                                         (layer["type"]))

            self.forward.append(aa)
            if i:
                self.forward[-1].link_from(self.forward[-2])
                self.forward[-1].link_attrs(self.forward[-2],
                                            ("input", "output"))
            else:
                self.forward[-1].link_from(self.loader)
                self.forward[-1].link_attrs(self.loader,
                                            ("input", "minibatch_data"))

        # Add Accumulator units
        self.accumulator = []
        for i in range(0, len(layers)):
            accum = accumulator.RangeAccumulator(self,
                                                 bars=root.accumulator.n_bars)
            self.accumulator.append(accum)
        self.accumulator[-1].link_from(self.forward[-1])
        self.accumulator[-1].link_attrs(self.forward[-1],
                                        ("input", "output"))

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(
            self, out_dirs=root.image_saver.out_dirs)
        self.image_saver.link_from(self.accumulator[-1])
        #self.image_saver.link_from(self.forward[-1])
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
            use_dynamic_alpha=root.decision.use_dynamic_alpha,
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
        for i in range(0, len(layers)):
            self.accumulator[i].reset_flag = ~self.decision.epoch_ended

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
            elif isinstance(self.forward[i], all2all.All2AllRELU):
                obj = gd.GDRELU(self, device=device)
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

        self.repeater.link_from(self.gd[0])

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
        self.plt_mx = []
        for i in range(0, len(layers)):
            self.decision.vectors_to_sync[self.gd[0].weights] = 1
            plt_mx = plotting_units.Weights2D(
                self, name="%s Layer Weights %s" % (i + 1, layers[i]["type"]),
                limit=root.weights_plotter.limit)
            self.plt_mx.append(plt_mx)
            self.plt_mx[i].input = self.gd[i].weights
            self.plt_mx[i].input_field = "v"
            if layers[i].get("n_kernels") is not None:
                self.plt_mx[i].get_shape_from = (
                    [self.forward[i].kx, self.forward[i].ky])
            if layers[i].get("layers") is not None:
                self.plt_mx[i].get_shape_from = self.forward[i].input
            self.plt_mx[i].link_from(self.decision)
            self.plt_mx[i].gate_block = ~self.decision.epoch_ended

        # Histogram plotter
        self.plt_hist = []
        for i in range(0, len(layers)):
            hist = plotting_units.Histogram(self, name="Histogram output %s" %
                                            (i + 1))
            self.plt_hist.append(hist)

        self.plt_hist[-1].link_from(self.decision)
        self.plt_hist[-1].input = self.accumulator[i].output
        self.plt_hist[-1].n_bars = self.accumulator[i].n_bars
        self.plt_hist[-1].x = self.accumulator[i].input
        self.plt_hist[-1].gate_block = ~self.decision.epoch_ended

        # MultiHistogram plotter
        self.plt_multi_hist = []
        for i in range(0, len(layers)):
            multi_hist = plotting_units.MultiHistogram(
                self, name="Histogram weights %s %s" % (i + 1,
                                                        layers[i]["type"]))
            self.plt_multi_hist.append(multi_hist)
            if layers[i].get("n_kernels") is not None:
                self.plt_multi_hist[i].link_from(self.decision)
                self.plt_multi_hist[i].hist_number = layers[i]["n_kernels"]
                self.plt_multi_hist[i].input = self.forward[i].weights
                end_epoch = ~self.decision.epoch_ended
                self.plt_multi_hist[i].gate_block = end_epoch
            if layers[i].get("layers") is not None:
                self.plt_multi_hist[i].link_from(self.decision)
                self.plt_multi_hist[i].hist_number = layers[i]["layers"]
                self.plt_multi_hist[i].input = self.forward[i].weights
                self.plt_multi_hist[i].gate_block = ~self.decision.epoch_ended

    def initialize(self, global_alpha, global_lambda, minibatch_size,
                   w_neg, device):
        self.loader.minibatch_maxsize = minibatch_size
        self.loader.w_neg = w_neg
        self.ev.device = device
        for g in self.gd:
            g.device = device
            g.global_alpha = global_alpha
            g.global_lambda = global_lambda
        for forward in self.forward:
            forward.device = device
        return super(Workflow, self).initialize()


def run(load, main):
    w_neg = None
    try:
        w, _ = load(Workflow, layers=root.channels_conv.layers)
        if root.channels_conv.export:
            tm = time.localtime()
            s = "%d.%02d.%02d_%02d.%02d.%02d" % (
                tm.tm_year, tm.tm_mon, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec)
            fnme = os.path.join(root.common.snapshot_dir,
                                "channels_workflow_%s" % s)
            try:
                w.export(fnme)
                logging.info("Exported successfully to %s.tar.gz" % (fnme))
            except:
                a, b, c = sys.exc_info()
                traceback.print_exception(a, b, c)
                logging.error("Error while exporting.")
            return
        if root.channels_conv.find_negative > 0:
            if type(w) != tuple or len(w) != 2:
                logging.error(
                    "Snapshot with weights and biases only "
                    "should be provided when find_negative is supplied. "
                    "Will now exit.")
                return
            w_neg = w
            raise IOError()
    except IOError:
        if root.channels_conv.export:
            logging.error("Valid snapshot should be provided if "
                          "export is True. Will now exit.")
            return
        if (root.channels_conv.find_negative > 0 and w_neg is None):
            logging.error("Valid snapshot should be provided if "
                          "find_negative supplied. Will now exit.")
            return
    fnme = (os.path.join(root.common.cache_dir, root.decision.snapshot_prefix)
            + ".txt")
    logging.info("Dumping file map to %s" % (fnme))
    fout = open(fnme, "w")
    file_map = w.loader.file_map
    for i in sorted(file_map.keys()):
        fout.write("%d\t%s\n" % (i, file_map[i]))
    fout.close()
    logging.info("Done")
    logging.info("Will execute workflow now")
    main(global_alpha=root.channels_conv.global_alpha,
         global_lambda=root.channels_conv.global_lambda,
         minibatch_size=root.loader.minibatch_size, w_neg=w_neg)