"""
Created on Aug 14, 2013

ImageLoader class.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from __future__ import division
import numpy
import os
from zope.interface import implementer

import veles.error as error
import veles.formats as formats
from veles.znicz.loader import (IFullBatchLoader, FullBatchLoader,
                                FullBatchLoaderMSE)


@implementer(IFullBatchLoader)
class ImageLoader(FullBatchLoader):
    """Loads images from multiple folders as full batch.

    Attributes:
        test_paths: list of paths with mask for test set,
                    for example: ["/tmp/\*.png"].
        validation_paths: list of paths with mask for validation set,
                          for example: ["/tmp/\*.png"].
        train_paths: list of paths with mask for train set,
                     for example: ["/tmp/\*.png"].
        target_paths: list of paths for target in case of MSE.
        target_by_lbl: dictionary of targets by lbl
                       in case of classification and MSE.

    Should be overriden in child class:
        get_label_from_filename()
        is_valid_filename()
    """
    def __init__(self, workflow, **kwargs):
        super(ImageLoader, self).__init__(workflow, **kwargs)
        self.test_paths = kwargs.get("test_paths")
        self.validation_paths = kwargs.get("validation_paths")
        self.train_paths = kwargs.get("train_paths")
        self.target_paths = kwargs.get("target_paths")
        self.grayscale = kwargs.get("grayscale", True)

    def init_unpickled(self):
        super(ImageLoader, self).init_unpickled()
        self.target_by_lbl = {}

    def from_image(self, fnme):
        """
        Loads data from image and normalizes it.

        Returns:
            :class:`numpy.ndarrayarray`: if there was one image in the file.
            tuple: `(data, labels)` if there were many images in the file
        """
        import scipy.ndimage
        a = scipy.ndimage.imread(fnme, flatten=self.grayscale)
        a = a.astype(numpy.float32)
        if self.normalize:
            formats.normalize(a)
        return a

    def get_label_from_filename(self, filename):
        """Returns label from filename.
        """
        pass

    def is_valid_filename(self, filename):
        return True

    def load_original(self, pathname):
        """Loads data from original files.
        """
        self.info("Loading from %s..." % (pathname))
        files = []
        for basedir, _, filelist in os.walk(pathname):
            for nme in filelist:
                fnme = "%s/%s" % (basedir, nme)
                if self.is_valid_filename(fnme):
                    files.append(fnme)
        files.sort()
        n_files = len(files)
        if not n_files:
            self.warning("No files fetched as %s" % (pathname))
            return [], []

        aa = None
        ll = []

        sz = -1
        this_samples = 0
        next_samples = 0
        for i in range(0, n_files):
            obj = self.from_image(files[i])
            if type(obj) == numpy.ndarray:
                a = obj
                if sz != -1 and a.size != sz:
                    raise error.BadFormatError("Found file with different "
                                               "size than first: %s", files[i])
                else:
                    sz = a.size
                lbl = self.get_label_from_filename(files[i])
                if lbl is not None:
                    if type(lbl) != int:
                        raise error.BadFormatError(
                            "Found non-integer label "
                            "with type %s for %s" % (str(type(ll)), files[i]))
                    ll.append(lbl)
                if aa is None:
                    sh = [n_files]
                    sh.extend(a.shape)
                    aa = numpy.zeros(sh, dtype=a.dtype)
                next_samples = this_samples + 1
            else:
                a, l = obj[0], obj[1]
                if len(a) != len(l):
                    raise error.BadFormatError(
                        "from_image() returned different number of samples "
                        "and labels.")
                if sz != -1 and a[0].size != sz:
                    raise error.BadFormatError(
                        "Found file with different sample size than first: %s",
                        files[i])
                else:
                    sz = a[0].size
                ll.extend(l)
                if aa is None:
                    sh = [n_files + len(l) - 1]
                    sh.extend(a[0].shape)
                    aa = numpy.zeros(sh, dtype=a[0].dtype)
                next_samples = this_samples + len(l)
            if aa.shape[0] < next_samples:
                aa = numpy.append(aa, a, axis=0)
            aa[this_samples:next_samples] = a
            self.total_samples += next_samples - this_samples
            this_samples = next_samples
        return (aa, ll)

    def load_data(self):
        data = None
        labels = []

        # Loading original data and labels.
        offs = 0
        i = -1
        for t in (self.test_paths, self.validation_paths, self.train_paths):
            i += 1
            if t is None or not len(t):
                continue
            for pathname in t:
                aa, ll = self.load_original(pathname)
                if not len(aa):
                    continue
                if len(ll):
                    if len(ll) != len(aa):
                        raise error.BadFormatError(
                            "Number of labels %d differs "
                            "from number of input images %d for %s" %
                            (len(ll), len(aa), pathname))
                    labels.extend(ll)
                elif len(labels):
                    raise error.BadFormatError("Not labels found for %s" %
                                               pathname)
                if data is None:
                    data = aa
                else:
                    data = numpy.append(data, aa, axis=0)
            self.class_lengths[i] = len(data) - offs
            offs = len(data)

        if len(labels):
            max_ll = max(labels)
            self.info("Labels are indexed from-to: %d %d" %
                      (min(labels), max_ll))
            self.original_labels.mem = numpy.array(labels, dtype=numpy.int32)

        # Loading target data and labels.
        if self.target_paths is not None:
            n = 0
            for pathname in self.target_paths:
                aa, ll = self.load_original(pathname)
                if len(ll):  # there are labels
                    for i, label in enumerate(ll):
                        self.target_by_lbl[label] = aa[i]
                else:  # assume that target order is the same as data
                    for a in aa:
                        self.target_by_lbl[n] = a
                        n += 1
            if n:
                if n != numpy.sum(self.class_lengths):
                    raise error.BadFormatError("Target samples count differs "
                                               "from data samples count.")
                self.original_labels.mem = numpy.arange(n, dtype=numpy.int32)

        self.original_data.mem = data


class ImageLoaderMSE(ImageLoader, FullBatchLoaderMSE):
    def load_data(self):
        super(ImageLoaderMSE, self).load_data()

        target = None
        for aa in self.target_by_lbl.values():
            sh = [len(self.original_data)]
            sh.extend(aa.shape)
            target = numpy.zeros(sh, dtype=aa.dtype)
            break
        if target is not None:
            for i, label in enumerate(self.original_labels.mem):
                target[i] = self.target_by_lbl[label]
            self.target_by_lbl.clear()
        self.original_targets.mem = target
