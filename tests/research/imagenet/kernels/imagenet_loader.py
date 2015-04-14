# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on July 30, 2014

ImageLoader class for imagenet

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


from __future__ import division
import os

import numpy
from zope.interface import implementer
import matplotlib.pyplot as plt
import cv2

from veles.loader.image import FullBatchFileImageLoader
import veles.error as error
from veles.znicz.loader import IFullBatchLoader


@implementer(IFullBatchLoader)
class ImgLoaderClassifier(FullBatchFileImageLoader):
    def __init__(self, workflow, **kwargs):
        super(ImgLoaderClassifier, self).__init__(workflow, **kwargs)
        path_mean_img = kwargs["path_mean_img"]
        bottom_conv1_size = 227
        self.mean_img = numpy.load(path_mean_img)
        self.mean_img = numpy.swapaxes(self.mean_img, 0, 1)
        self.mean_img = numpy.swapaxes(self.mean_img, 1, 2)
        self.mean_img = cv2.resize(self.mean_img, (bottom_conv1_size,
                                                   bottom_conv1_size))

    def get_image_data(self, key, new_size=227):
        """Loads one image image.
        Args:
            fnme(string): name of file with image
            new_size(int): image size for top bottom conv1
        Returns:
            numpy array
        """
        img = plt.imread(key)
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
