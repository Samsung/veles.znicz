"""
Created on Aug 14, 2013

ImageLoader class.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from __future__ import division
import cv2
from itertools import chain, zip_longest
import logging
import numpy
import os
from PIL import Image
from psutil import phymem_usage
from zope.interface import implementer, Interface

import veles.error as error
from veles.external.progressbar import ProgressBar, Percentage, Bar
import veles.memory as memory
from veles.znicz.loader.base import CLASS_NAME, TARGET
from veles.znicz.loader.fullbatch import (IFullBatchLoader, FullBatchLoader,
                                          FullBatchLoaderMSE)


MODE_COLOR_MAP = {
    "1": "GRAY",
    "L": "GRAY",
    "P": "RGB",
    "RGB": "RGB",
    "RGBA": "RGBA",
    "CMYK": "RGB",
    "YCbCr": "YCR_CB",
    "I": "GRAY",
    "F": "GRAY",
}


class IImageLoader(Interface):
    def get_image_label(key):
        """Retrieves label for the specified key.
        """

    def get_image_info(key):
        """
        Return an iterable of tuples, each tuple is a pair
        (size, color space). Size must be in OpenCV order (first y, then x),
        color space must be supported by OpenCV (COLOR_*).
        """

    def get_image_data(key):
        """Returns the image data associated with the specified key.
        """

    def get_keys(index):
        """
        Return a list of image keys to process for the specified class index.
        """


@implementer(IFullBatchLoader)
class FullBatchImageLoader(FullBatchLoader):
    """Loads images into fully in-memory batch.

    Attributes:
        color_space: the color space to which to convert images. Can be any of
                     the values supported by OpenCV, e.g., GRAY or HSV.
        normalize: True if image data must be normalized; otherwise, False.
        source_dtype: dtype to work with during various image operations.
        shape: image shape (tuple) - set after initialize().

    Must be overriden in child class:
        get_image_label()
        get_image_info()
        get_image_data()
        get_keys()
    """

    def __init__(self, workflow, **kwargs):
        super(FullBatchImageLoader, self).__init__(workflow, **kwargs)
        self.color_space = kwargs.get("color_space", "RGB")
        self.normalize = kwargs.get("normalize", True)
        self.source_dtype = numpy.float32
        self.shape = tuple()
        self.verify_interface(IImageLoader)

    def preprocess_image(self, data, key):
        _, color = self.get_image_info(key)
        if color != self.color_space:
            method = getattr(
                cv2, "COLOR_%s2%s" % (color, self.color_space), None)
            if method is None:
                data = cv2.cvtColor(data, getattr(cv2, "COLOR_%s2BGR" % color))
                method = getattr(cv2, "COLOR_BGR2%s" % self.color_space)
            data = cv2.cvtColor(data, method)
        if self.normalize:
            # TODO(v.markovtsev): implement normalization
            memory.normalize(data)
        return data

    def load_keys(self, keys, pbar, data, labels):
        """Loads data from the specified keys.
        """
        index = 0
        has_labels = False
        for key in keys:
            obj = self.preprocess_image(self.get_image_data(key), key)
            if not isinstance(obj, numpy.ndarray):
                objs, obj_labels = obj
            else:
                objs = (obj,)
                obj_labels = (self.get_image_label(key),)
            for obj, label in zip_longest(objs, obj_labels):
                if label is not None:
                    has_labels = True
                    assert isinstance(label, int), \
                        "Got non-integer label %s of type %s for %s" % (
                            label, label.__class__, key)
                if has_labels and label is None:
                    raise error.BadFormatError(
                        "%s does not have a label, but others do" % key)
                data[index] = obj
                labels[index] = label
                index += 1
                if pbar is not None:
                    pbar.inc()
        return has_labels

    def load_data(self):
        keys = [[], [], []]
        # First pass
        for index, class_name in enumerate(CLASS_NAME):
            class_keys = self.get_keys(index)
            keys[index].extend(class_keys)
            self.class_lengths[index] += len(class_keys)
            keys[index].sort()
        if self.shape == tuple():
            raise error.BadFormatError("Image shape was not initialized in "
                                       "get_keys()")

        # Allocate data
        overall = sum(self.class_lengths)
        self.info("Found %d samples of shape %s (%d TEST, %d VALIDATION, "
                  "%d TRAIN)", overall, self.shape, *self.class_lengths)
        required_mem = overall * numpy.prod(self.shape) * numpy.dtype(
            self.source_dtype).itemsize
        if phymem_usage().free < required_mem:
            gb = 1.0 / (1000 * 1000 * 1000)
            self.critical("Not enough memory (free %.3f Gb, required %.3f Gb)",
                          phymem_usage().free * gb, required_mem * gb)
            raise MemoryError("Not enough memory")
        # Real allocation will still happen during the second pass
        self.original_data.mem = data = numpy.zeros(
            (overall,) + self.shape, dtype=self.source_dtype)
        self.original_labels.mem = labels = numpy.zeros(
            overall, dtype=numpy.int32)

        # Second pass
        pbar = ProgressBar(term_width=17, maxval=overall,
                           widgets=["Loading %d images " % overall,
                                    Percentage(), ' ', Bar()],
                           log_level=logging.INFO)
        pbar.start()
        offset = 0
        has_labels = []
        for class_keys in keys:
            if len(class_keys) > 0:
                has_labels.append(self.load_keys(
                    class_keys, pbar, data[offset:], labels[offset:]))
                offset += len(class_keys)
        pbar.finish()

        # Delete labels mem if no labels was extracted
        if numpy.prod(has_labels) == 0 and sum(has_labels) > 0:
            raise error.BadFormatError(
                "Some classes do not have labels while other do")
        if sum(has_labels) == 0:
            self.original_labels.mem = None


class FullBatchImageLoaderMSE(FullBatchImageLoader, FullBatchLoaderMSE):
    """
    MSE modification of FullBatchImageLoader class.
    """

    def load_data(self):
        super(FullBatchImageLoaderMSE, self).load_data()

        keys = self.get_keys(TARGET)
        keys.sort()
        length = len(keys)
        targets = numpy.zeros((length,) + self.shape, dtype=self.source_dtype)
        target_labels = numpy.zeros(length, dtype=numpy.int32)
        has_labels = self.load_keys(keys, None, targets, target_labels)
        if not has_labels:
            if self.original_labels.mem is not None:
                raise error.BadFormatError(
                    "Targets do not have labels, but the classes do")
            if sum(self.class_lengths) != length:
                raise error.BadFormatError(
                    "Number of class samples %d differs from the number of "
                    "targets %d" % (sum(self.class_lengths), length))
            # Associate targets with classes by sequence order => NOP
            self.original_targets.mem = targets
            return
        if self.original_labels.mem is None:
            raise error.BadFormatError(
                "Targets have labels, but the classes do not")
        if len(set(target_labels)) < length:
            raise error.BadFormatError("Some targets have duplicate labels")
        diff = set(self.original_labels).difference(target_labels)
        if len(diff) > 0:
            raise error.BadFormatError(
                "Labels %s do not have corresponding targets" % diff)
        self.original_targets.mem = numpy.zeros(
            (self.original_labels.shape[0],) + self.shape, self.source_dtype)
        target_mapping = {target_labels[i]: i for i in range(length)}
        for i, label in enumerate(self.original_labels):
            self.original_targets[i] = targets[target_mapping[label]]


class IFileImageLoader(Interface):
    def get_label_from_filename(filename):
        """Retrieves label for the specified file path.
        """

    def is_valid_filename(filename):
        """Filters the file names. Return True if the specified file path must
-        be included, otherwise, False.
        """


@implementer(IImageLoader)
class FullBatchFileImageLoader(FullBatchImageLoader):
    """Loads images from multiple folders as full batch.

    Attributes:
        test_paths: list of paths with mask for test set,
                    for example: ["/tmp/\*.png"].
        validation_paths: list of paths with mask for validation set,
                          for example: ["/tmp/\*.png"].
        train_paths: list of paths with mask for train set,
                     for example: ["/tmp/\*.png"].

    Must be overriden in child class:
        get_label_from_filename()
        is_valid_filename()
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchFileImageLoader, self).__init__(workflow, **kwargs)
        self.test_paths = kwargs.get("test_paths")
        self.validation_paths = kwargs.get("validation_paths")
        self.train_paths = kwargs.get("train_paths")
        self.verify_interface(IFileImageLoader)

    def _check_paths(self, paths):
        if not hasattr(paths, "__iter__"):
            raise TypeError("Paths must be iterable, e.g., a list instance")

    @property
    def test_paths(self):
        return self._test_paths

    @test_paths.setter
    def test_paths(self, value):
        self._check_paths(value)
        self._test_paths = value

    @property
    def validation_paths(self):
        return self._validation_paths

    @validation_paths.setter
    def validation_paths(self, value):
        self._check_paths(value)
        self._validation_paths = value

    @property
    def train_paths(self):
        return self._train_paths

    @train_paths.setter
    def train_paths(self, value):
        self._check_paths(value)
        self._train_paths = value

    def get_image_label(self, key):
        return self.get_label_from_filename(key)

    def get_image_info(self, key):
        """
        :param key: The full path to the analysed image.
        :return: iterable of tuples, each tuple is a pair
        (image size, number of channels).
        """
        try:
            img = Image.open(key)
            return (tuple(reversed(img.size)), MODE_COLOR_MAP[img.mode]),
        except:
            # Unable to read the image with PIL. Fall back to slow OpenCV
            # method which reads the whole image.
            img = cv2.imread(key, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise error.BadFormatError("Unable to read %s" % key)
            return (img.shape[:2] + (3,), "BGR"),

    def get_image_data(self, key):
        """
        Loads data from image and normalizes it.

        Returns:
            :class:`numpy.ndarrayarray`: if there was one image in the file.
            tuple: `(data, labels)` if there were many images in the file
        """
        try:
            img = Image.open(key)
            if img.mode in ("P", "CMYK"):
                return numpy.array(img.convert("RGB"), dtype=self.source_dtype)
            else:
                return numpy.array(img, dtype=self.source_dtype)
        except:
            img = cv2.imread(key)
            if img is None:
                raise error.BadFormatError("Unable to read %s" % key)
            return img.astype(self.source_dtype)

    def scan_files(self, pathname):
        self.info("Scanning %s..." % pathname)
        files = []
        for basedir, _, filelist in os.walk(pathname):
            for name in filelist:
                full_name = os.path.join(basedir, name)
                if self.is_valid_filename(full_name):
                    files.append(full_name)
        if not len(files):
            self.warning("No files were taken from %s" % pathname)
            return [], []

        # First pass: get the final list of files and shape
        self.debug("Analyzing %d images in %s", len(files), pathname)
        uniform_files = []
        for file in files:
            infos = self.get_image_info(file)
            for size, channels in infos:
                shape = size + tuple(channels)
                if self.shape != tuple() and shape != self.shape:
                    self.warning("%s has the different shape %s (expected %s)",
                                 file, shape, self.shape)
                else:
                    self.shape = shape
                    uniform_files.append(file)
        return uniform_files

    def get_keys(self, index):
        paths = (self.test_paths, self.validation_paths,
                 self.train_paths)[index]
        if paths is None:
            return []
        return list(chain.from_iterable(self.scan_files(p) for p in paths))


class FullBatchFileImageLoaderMSE(FullBatchFileImageLoader,
                                  FullBatchImageLoaderMSE):
    """
    MSE modification of  FullBatchFileImageLoader class.
    Attributes:
        target_paths: list of paths for target in case of MSE.
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchFileImageLoaderMSE, self).__init__(workflow, **kwargs)
        self.target_paths = kwargs["target_paths"]

    @property
    def target_paths(self):
        return self._target_paths

    @target_paths.setter
    def target_paths(self, value):
        self._check_paths(value)
        self._target_paths = value

    def get_keys(self, index):
        if index != TARGET:
            return super(FullBatchFileImageLoaderMSE, self).get_keys(index)
        return list(chain.from_iterable(
            self.scan_files(p) for p in self.target_paths))
