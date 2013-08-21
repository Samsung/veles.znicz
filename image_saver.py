"""
Created on Aug 20, 2013

ImageSaver unit.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import units
import scipy.misc
import os
import glob


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

    Remarks:
        output and target should be both None or both not None.
    """
    def __init__(self, out_dirs=[".", ".", "."]):
        super(ImageSaver, self).__init__()
        self.out_dirs = out_dirs
        self.input = None  # formats.Vector()
        self.output = None  # formats.Vector()
        self.target = None  # formats.Vector()
        self.indexes = None  # formats.Vector()
        self.labels = None  # formats.Vector()
        self.minibatch_class = None  # [0]
        self.minibatch_size = None  # [0]
        self.this_save_time = [0]
        self.last_save_time = 0

    def run(self):
        self.input.sync()
        if self.output != None:
            self.output.sync()
        self.indexes.sync()
        self.labels.sync()
        if self.last_save_time < self.this_save_time[0]:
            self.last_save_time = self.this_save_time[0]
            for dirnme in self.out_dirs:
                i = 0
                while True:
                    j = dirnme.find("/", i)
                    if j <= i:
                        break
                    try:
                        os.mkdir(dirnme[:j - 1])
                    except OSError:
                        pass
                    i = j + 1
                files = glob.glob("%s/*.png" % (dirnme))
                for file in files:
                    try:
                        os.unlink(file)
                    except OSError:
                        pass
        xyt = None
        x = None
        y = None
        t = None
        for i in range(0, self.minibatch_size[0]):
            x = self.input.v[i]
            if self.output != None and self.target != None:
                y = self.output.v[i]
                t = self.target.v[i]
                y = y.reshape(t.shape)
            idx = self.indexes.v[i]
            lbl = self.labels.v[i]
            mse = numpy.linalg.norm(t - y) / x.size
            if xyt == None:
                n_rows = x.shape[0]
                n_cols = x.shape[1]
                if y != None:
                    n_rows += y.shape[0]
                    n_cols = max(n_cols, y.shape[1])
                if y != t:
                    n_rows += t.shape[0]
                    n_cols = max(n_cols, t.shape[1])
                xyt = numpy.empty([n_rows, n_cols], dtype=x.dtype)
            xyt[:] = 0
            offs = (xyt.shape[1] - x.shape[1]) >> 1
            xyt[:x.shape[0], offs:offs + x.shape[1]] = x[:, :]
            img = xyt[:x.shape[0], offs:offs + x.shape[1]]
            img *= -1.0
            img += 1.0
            img *= 127.5
            numpy.clip(img, 0, 255, img)
            if y != None:
                offs = (xyt.shape[1] - y.shape[1]) >> 1
                xyt[x.shape[0]:x.shape[0] + y.shape[0],
                    offs:offs + y.shape[1]] = y[:, :]
                img = xyt[x.shape[0]:x.shape[0] + y.shape[0],
                          offs:offs + y.shape[1]]
                img *= -1.0
                img += 1.0
                img *= 127.5
                numpy.clip(img, 0, 255, img)
            if y != t:
                offs = (xyt.shape[1] - t.shape[1]) >> 1
                xyt[x.shape[0] + y.shape[0]:, offs:offs + t.shape[1]] = t[:, :]
                img = xyt[x.shape[0] + y.shape[0]:, offs:offs + t.shape[1]]
                img *= -1.0
                img += 1.0
                img *= 127.5
                numpy.clip(img, 0, 255, img)
            fnme = "%s/%.6f_%d_%d.png" % (
                self.out_dirs[self.minibatch_class[0]], mse, lbl, idx)
            scipy.misc.imsave(fnme, xyt.astype(numpy.uint8))
