"""
Created on Aug 20, 2013

ImageSaver unit.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import glob
import numpy
import os

import veles.formats as formats
from veles.config import root
import veles.units as units


class ImageSaver(units.Unit):
    """Saves input to pngs in the supplied directory.

    Will remove all existing png files in the supplied directory.

    Attributes:
        out_dirs: output directories by minibatch_class where to save png.
        input: batch with input samples.
        output: batch with corresponding output samples (may be None).
        target: batch with corresponding target samples (may be None).
        indexes: sample indexes.
        labels: sample labels.
        max_idx: indexes of element with maximum value for each sample.

    Remarks:
        if max_idx is not None:
            Softmax classifier is assumed and only failed samples
            will be saved.
        else:
            MSE task is assumed and output and target
            should be None or not None both simultaneously.
    """
    def __init__(self, workflow, **kwargs):
        out_dirs = kwargs.get("out_dirs")
        limit = kwargs.get("limit", 100)
        yuv = kwargs.get("yuv", False)
        if out_dirs is None:
            out_dirs = ["%s/tmpimg/test" % (root.common.cache_dir),
                        "%s/tmpimg/validation" % (root.common.cache_dir),
                        "%s/tmpimg/train" % (root.common.cache_dir)]
        kwargs["out_dirs"] = out_dirs
        kwargs["limit"] = limit
        kwargs["yuv"] = yuv
        super(ImageSaver, self).__init__(workflow, **kwargs)
        self.out_dirs = out_dirs
        self.input = None  # formats.Vector()
        self.output = None  # formats.Vector()
        self.target = None  # formats.Vector()
        self.indexes = None  # formats.Vector()
        self.labels = None  # formats.Vector()
        self.max_idx = None  # formats.Vector()
        self.minibatch_class = None  # [0]
        self.minibatch_size = None  # [0]
        self.this_save_time = [0]
        self.last_save_time = 0
        self.limit = limit
        self.n_saved = [0, 0, 0]
        self.yuv = [1 if yuv else 0]

    def as_image(self, x):
        if len(x.shape) == 2:
            return x.reshape(x.shape[0], x.shape[1], 1)
        if len(x.shape) == 3:
            if x.shape[2] == 3:
                return x
            if x.shape[0] == 3:
                xx = numpy.empty([x.shape[1], x.shape[2], 3],
                                 dtype=x.dtype)
                xx[:, :, 0:1] = x[0:1, :, :].reshape(
                    x.shape[1], x.shape[2], 1)[:, :, 0:1]
                xx[:, :, 1:2] = x[1:2, :, :].reshape(
                    x.shape[1], x.shape[2], 1)[:, :, 0:1]
                xx[:, :, 2:3] = x[2:3, :, :].reshape(
                    x.shape[1], x.shape[2], 1)[:, :, 0:1]
                return xx
        return x.ravel()

    def run(self):
        import scipy.misc
        self.input.map_read()
        if self.output is not None:
            self.output.map_read()
        self.indexes.map_read()
        self.labels.map_read()
        if self.max_idx is not None:
            self.max_idx.map_read()
        if self.last_save_time < self.this_save_time[0]:
            self.last_save_time = self.this_save_time[0]
            for i in range(len(self.n_saved)):
                self.n_saved[i] = 0
            for dirnme in self.out_dirs:
                try:
                    os.makedirs(dirnme, mode=0o775, exist_ok=True)
                except OSError:
                    pass
                files = glob.glob("%s/*.png" % (dirnme))
                for file in files:
                    try:
                        os.unlink(file)
                    except OSError:
                        pass
        if self.n_saved[self.minibatch_class[0]] >= self.limit:
            return
        xyt = None
        x = None
        y = None
        t = None
        im = 0
        for i in range(0, self.minibatch_size[0]):
            x = self.as_image(self.input.v[i])
            idx = self.indexes.v[i]
            lbl = self.labels.v[i]
            if self.max_idx is not None:
                im = self.max_idx[i]
                if im == lbl:
                    continue
                y = self.output.v[i]
            if (self.max_idx is None and
                    self.output is not None and self.target is not None):
                y = self.as_image(self.output.v[i])
                t = self.as_image(self.target.v[i])
                y = y.reshape(t.shape)
            if self.max_idx is None and y is not None:
                mse = numpy.linalg.norm(t - y) / x.size
            if xyt is None:
                n_rows = x.shape[0]
                n_cols = x.shape[1]
                if (self.max_idx is None and y is not None and
                        len(y.shape) != 1):
                    n_rows += y.shape[0]
                    n_cols = max(n_cols, y.shape[1])
                if (self.max_idx is None and t is not None and
                        len(t.shape) != 1 and self.input != self.target):
                    n_rows += t.shape[0]
                    n_cols = max(n_cols, t.shape[1])
                xyt = numpy.empty([n_rows, n_cols, x.shape[2]], dtype=x.dtype)
            xyt[:] = 0
            offs = (xyt.shape[1] - x.shape[1]) >> 1
            xyt[:x.shape[0], offs:offs + x.shape[1]] = x[:, :]
            img = xyt[:x.shape[0], offs:offs + x.shape[1]]
            #img *= -1.0
            img += 1.0
            img *= 127.5
            numpy.clip(img, 0, 255, img)
            if self.max_idx is None and y is not None and len(y.shape) != 1:
                offs = (xyt.shape[1] - y.shape[1]) >> 1
                xyt[x.shape[0]:x.shape[0] + y.shape[0],
                    offs:offs + y.shape[1]] = y[:, :]
                img = xyt[x.shape[0]:x.shape[0] + y.shape[0],
                          offs:offs + y.shape[1]]
                #img *= -1.0
                img += 1.0
                img *= 127.5
                numpy.clip(img, 0, 255, img)
            if (self.max_idx is None and t is not None and
                    len(t.shape) != 1 and self.input != self.target):
                offs = (xyt.shape[1] - t.shape[1]) >> 1
                xyt[x.shape[0] + y.shape[0]:, offs:offs + t.shape[1]] = t[:, :]
                img = xyt[x.shape[0] + y.shape[0]:, offs:offs + t.shape[1]]
                #img *= -1.0
                img += 1.0
                img *= 127.5
                numpy.clip(img, 0, 255, img)
            if self.max_idx is None:
                fnme = "%s/%.6f_%d_%d.png" % (
                    self.out_dirs[self.minibatch_class[0]], mse, lbl, idx)
            else:
                fnme = "%s/%d_as_%d.%.0fpt.%d.png" % (
                    self.out_dirs[self.minibatch_class[0]], lbl, im, y[im],
                    idx)
            img = xyt
            if img.shape[2] == 1:
                img = img.reshape(img.shape[0], img.shape[1])
            try:
                scipy.misc.imsave(fnme, formats.norm_image(img, self.yuv[0]))
            except OSError:
                self.error("Could not save image to %s" % (fnme))
            self.n_saved[self.minibatch_class[0]] += 1
            if self.n_saved[self.minibatch_class[0]] >= self.limit:
                return
