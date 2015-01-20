"""
Created on Aug 14, 2013

ImageLoader class.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from __future__ import division
from itertools import zip_longest
import logging
import numpy
import os
from PIL import Image
from psutil import phymem_usage
from zope.interface import implementer, Interface

import veles.error as error
from veles.external.progressbar import ProgressBar, Percentage, Bar
import veles.memory as formats
from veles.znicz.loader.fullbatch import (IFullBatchLoader, FullBatchLoader,
                                          FullBatchLoaderMSE)


MODE_CHANNEL_MAP = {
    "1": 1,
    "L": 1,
    "P": 1,
    "RGB": 3,
    "RGBA": 4,
    "CMYK": 4,
    "YCbCr": 3,
    "I": 1,
    "F": 1,
}


class IImageLoader(Interface):
    def get_label_from_filename(filename):
        """Retrieves label for the specified file path.
        """

    def is_valid_filename(filename):
        """Filters the file names. Return True if the specified file path must
        be included, otherwise, False.
        """


@implementer(IFullBatchLoader)
class FullBatchImageLoader(FullBatchLoader):
    """Loads images from multiple folders as full batch.

    Attributes:
        test_paths: list of paths with mask for test set,
                    for example: ["/tmp/\*.png"].
        validation_paths: list of paths with mask for validation set,
                          for example: ["/tmp/\*.png"].
        train_paths: list of paths with mask for train set,
                     for example: ["/tmp/\*.png"].
        target_paths: list of paths for target in case of MSE.
        source_dtype: dtype to work with during various image operations.
        shape: image shape (tuple) - set after initialize().

    Must be overriden in child class:
        get_label_from_filename()
        is_valid_filename()
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchImageLoader, self).__init__(workflow, **kwargs)
        self.test_paths = kwargs.get("test_paths")
        self.validation_paths = kwargs.get("validation_paths")
        self.train_paths = kwargs.get("train_paths")
        self.grayscale = kwargs.get("grayscale", True)
        self.source_dtype = numpy.float32
        self.shape = tuple()
        self.verify_interface(IImageLoader)

    def init_unpickled(self):
        super(FullBatchImageLoader, self).init_unpickled()
        self.target_by_lbl = {}

    def get_image_info(self, file_name):
        """
        :param file_name: The full path to the analysed image.
        :return: iterable of tuples, each tuple is a pair
        (image size, number of channels).
        """
        img = Image.open(file_name)
        return tuple((tuple(reversed(img.size)),
                      MODE_CHANNEL_MAP[img.mode] if not self.grayscale else 1))

    def get_image_data(self, file_name):
        """
        Loads data from image and normalizes it.

        Returns:
            :class:`numpy.ndarrayarray`: if there was one image in the file.
            tuple: `(data, labels)` if there were many images in the file
        """
        img = Image.open(file_name)
        if self.grayscale:
            img = img.convert('F')
        data = numpy.array(img, dtype=numpy.float32)
        if self.normalize:
            formats.normalize(data)
        return data

    def scan_files(self, pathname):
        self.info("Loading from %s..." % pathname)
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

    def load_files(self, files, pbar, data, labels):
        """Loads data from original files.
        """
        # Second pass: load the actual data
        index = 0
        has_labels = False
        for file in files:
            obj = self.get_image_data(file)
            if not isinstance(obj, numpy.ndarray):
                objs, obj_labels = obj
            else:
                objs = (obj,)
                obj_labels = (self.get_label_from_filename(file),)
            for obj, label in zip_longest(objs, obj_labels):
                if label is not None:
                    has_labels = True
                    assert isinstance(label, int), \
                        "Got non-integer label %s of type %s for %s" % (
                            label, label.__class__, file)
                if has_labels and label is None:
                    raise error.BadFormatError(
                        "%s does not have a label, but others do" % file)
                data[index] = obj
                labels[index] = label
                index += 1
                if pbar is not None:
                    pbar.inc()
        return has_labels

    def load_data(self):
        files = [[], [], []]
        # First pass
        for index, path in enumerate((self.test_paths, self.validation_paths,
                                      self.train_paths)):
            if not path:
                continue
            for pathname in path:
                class_files = self.scan_files(pathname)
                files[index].extend(class_files)
                self.class_lengths[index] += len(class_files)
            files[index].sort()

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
        for class_files in files:
            if len(class_files) > 0:
                has_labels.append(self.load_files(
                    class_files, pbar, data[offset:], labels[offset:]))
                offset += len(class_files)
        pbar.finish()

        # Delete labels mem if no labels was extracted
        if numpy.prod(has_labels) == 0 and sum(has_labels) > 0:
            raise error.BadFormatError(
                "Some classes do not have labels while other do")
        if sum(has_labels) == 0:
            self.original_labels.mem = None


class FullBatchImageLoaderMSE(FullBatchImageLoader, FullBatchLoaderMSE):
    """
    MSE modification of ImageLoader class.
    Attributes:
        target_paths: list of paths for target in case of MSE.
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchImageLoaderMSE, self).__init__(workflow, **kwargs)
        self.target_paths = kwargs["target_paths"]

    def load_data(self):
        super(FullBatchImageLoaderMSE, self).load_data()

        files = []
        for pathname in self.target_paths:
            files.extend(self.scan_files(pathname))
        files.sort()
        length = len(files)
        targets = numpy.zeros((length,) + self.shape, dtype=self.source_dtype)
        target_labels = numpy.zeros(length, dtype=numpy.int32)
        has_labels = self.load_files(files, None, targets, target_labels)
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
