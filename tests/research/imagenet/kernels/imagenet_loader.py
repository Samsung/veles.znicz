"""
Created on July 30, 2014

ImageLoader class for imagenet

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""
from __future__ import division
from veles.znicz.loader.image import FullBatchImageLoader
import numpy
import os
from zope.interface import implementer
import veles.error as error
from veles.znicz.loader import IFullBatchLoader
import matplotlib.pyplot as plt
import cv2


@implementer(IFullBatchLoader)
class ImgLoaderClassifier(FullBatchImageLoader):
    def __init__(self, workflow, **kwargs):
        super(ImgLoaderClassifier, self).__init__(workflow, **kwargs)
        path_mean_img = kwargs["path_mean_img"]
        bottom_conv1_size = 227
        self.mean_img = numpy.load(path_mean_img)
        self.mean_img = numpy.swapaxes(self.mean_img, 0, 1)
        self.mean_img = numpy.swapaxes(self.mean_img, 1, 2)
        self.mean_img = cv2.resize(self.mean_img, (bottom_conv1_size,
                                                   bottom_conv1_size))

    def _update_total_samples(self):
        """Fills self.class_offsets from self.class_lengths.
        """
        total_samples = 0
        for i, n in enumerate(self.class_lengths):
            total_samples += n
            self.class_offsets[i] = total_samples
        self.total_samples = total_samples

    def get_image_data(self, file_name, new_size=227):
        """Loads one image image.
        Args:
            fnme(string): name of file with image
            new_size(int): image size for top bottom conv1
        Returns:
            numpy array
        """
        img = plt.imread(file_name)
        img = img[:, :, ::-1]
        img = cv2.resize(img, (227, 227))
        img = img - self.mean_img
        return img

    def load_original(self, pathname):
        """
        Loads data from original files.
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
            obj = self.get_image_data(files[i])
            if type(obj) == numpy.ndarray:
                wfl = obj
                if sz != -1 and wfl.size != sz:
                    raise error.BadFormatError("Found file with different "
                                               "size than first: %s", files[i])
                else:
                    sz = wfl.size
                lbl = self.get_label_from_filename(files[i])
                if lbl is not None:
                    if type(lbl) != int:
                        raise error.BadFormatError(
                            "Found non-integer label "
                            "with type %s for %s" % (str(type(ll)), files[i]))
                    ll.append(lbl)
                if aa is None:
                    sh = [n_files]
                    sh.extend(wfl.shape)
                    aa = numpy.zeros(sh, dtype=wfl.dtype)
                next_samples = this_samples + 1
            else:
                wfl, l = obj[0], obj[1]
                if len(wfl) != len(l):
                    raise error.BadFormatError(
                        "from_image() returned different number of samples "
                        "and labels.")
                if sz != -1 and wfl[0].size != sz:
                    raise error.BadFormatError(
                        "Found file with different sample size than first: %s",
                        files[i])
                else:
                    sz = wfl[0].size
                ll.extend(l)
                if aa is None:
                    sh = [n_files + len(l) - 1]
                    sh.extend(wfl[0].shape)
                    aa = numpy.zeros(sh, dtype=wfl[0].dtype)
                next_samples = this_samples + len(l)
            if aa.shape[0] < next_samples:
                aa = numpy.append(aa, wfl, axis=0)
            aa[this_samples:next_samples] = wfl
            self.total_samples += next_samples - this_samples
            this_samples = next_samples
        return (aa, ll)
