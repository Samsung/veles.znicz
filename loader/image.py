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
from psutil import virtual_memory
from zope.interface import implementer, Interface
from veles.compat import from_none

import veles.error as error
from veles.external.progressbar import ProgressBar, Percentage, Bar
import veles.memory as memory
from veles.znicz.loader.base import (
    CLASS_NAME, TARGET, ILoader, Loader, LoaderMSEMixin)
from veles.znicz.loader.fullbatch import (
    IFullBatchLoader, FullBatchLoader, FullBatchLoaderMSEMixin, DTYPE)


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

COLOR_CHANNELS_MAP = {
    "RGB": 3,
    "BGR": 3,
    "GRAY": 1,
    "HSV": 3,
    "YCR_CB": 3,
    "RGBA": 4,
    "BGRA": 4,
    "LAB": 3,
    "LUV": 3,
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


@implementer(ILoader)
class ImageLoader(Loader):
    """Base class for all image loaders. It is generally used for loading large
    datasets.

    Attributes:
        color_space: the color space to which to convert images. Can be any of
                     the values supported by OpenCV, e.g., GRAY or HSV.
        source_dtype: dtype to work with during various image operations.
        shape: image shape (tuple) - set after initialize().

     Must be overriden in child classes:
        get_image_label()
        get_image_info()
        get_image_data()
        get_keys()
    """

    def __init__(self, workflow, **kwargs):
        super(ImageLoader, self).__init__(workflow, **kwargs)
        self.color_space = kwargs.get("color_space", "RGB")
        self.source_dtype = numpy.float32
        self._shape = tuple()
        self._has_labels = False
        self.class_keys = [[], [], []]
        self.verify_interface(IImageLoader)
        self._restored_from_pickle = False

    def __setstate__(self, state):
        super(ImageLoader, self).__setstate__(state)
        self.info("Scanning for changes...")
        for keys in self.class_keys:
            for key in keys:
                (size, _), = self.get_image_info(key)
                if size != self.shape[:2]:
                    raise error.BadFormatError(
                        "%s changed the size (now %s, was %s)" %
                        (key, size, self.shape[:2]))
        self._restored_from_pickle = True

    @Loader.shape.getter
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if value is None:
            raise ValueError("shape must not be None")
        if not isinstance(value, tuple):
            raise TypeError("shape must be a tuple (got %s)" % (value,))
        if len(value) not in (2, 3):
            raise ValueError("len(shape) must be equal to 2 or 3 (got %s)" %
                             (value,))
        for i, d in enumerate(value):
            if not isinstance(d, int):
                raise TypeError("shape[%d] is not an integer (= %s)" % (i, d))
            if d < 1:
                raise ValueError("shape[%d] < 1 (= %s)" % (i, d))
        self._shape = value[:2]
        if self.channels_number > 1:
            self._shape += (self.channels_number,)

    @property
    def has_labels(self):
        """
        This is set after initialize() (particularly, after load_data()).
        """
        return self._has_labels

    @property
    def channels_number(self):
        return COLOR_CHANNELS_MAP[self.color_space]

    def preprocess_image(self, data, key):
        (_, color), = self.get_image_info(key)
        if color != self.color_space:
            method = getattr(
                cv2, "COLOR_%s2%s" % (color, self.color_space), None)
            if method is None:
                aux_method = getattr(cv2, "COLOR_%s2BGR" % color)
                try:
                    data = cv2.cvtColor(data, aux_method)
                except cv2.error as e:
                    self.error("Failed to perform '%s' conversion", aux_method)
                    raise from_none(e)
                method = getattr(cv2, "COLOR_BGR2%s" % self.color_space)
            try:
                data = cv2.cvtColor(data, method)
            except cv2.error as e:
                self.error("Failed to perform '%s' conversion", method)
                raise from_none(e)
        if self.normalize:
            # TODO(v.markovtsev): implement normalization, incl. Loader
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
                if obj.shape[:2] != self.shape[:2]:
                    self.debug("Ignored %s (label %s): shape %s",
                               key, label, obj.shape[:2])
                    continue
                if data is not None:
                    data[index] = obj
                if labels is not None:
                    labels[index] = label
                index += 1
                if pbar is not None:
                    pbar.inc()
        return has_labels

    def initialize(self, device, **kwargs):
        super(ImageLoader, self).initialize(device, **kwargs)
        self._restored_from_pickle = False

    def load_data(self):
        if self._restored_from_pickle:
            return
        for keys in self.class_keys:
            del keys[:]
        for index, class_name in enumerate(CLASS_NAME):
            keys = self.get_keys(index)
            self.class_keys[index].extend(keys)
            self.class_lengths[index] += len(keys)
            self.class_keys[index].sort()
        if self.shape == tuple():
            raise error.BadFormatError(
                "Image shape was not initialized in get_keys()")

        # Perform a quick (unreliable) test to determine if we have labels
        keys = []
        for i in range(3):
            keys = self.class_keys[i]
            if len(keys) > 0:
                break
        assert len(keys) > 0
        data = numpy.zeros((1,) + self.shape, dtype=self.source_dtype)
        labels = numpy.zeros((1,), dtype=Loader.LABEL_DTYPE)
        self._has_labels = self.load_keys(
            (keys[numpy.random.randint(len(keys))],), None, data, labels)

    def create_minibatches(self):
        self.minibatch_data.reset()
        self.minibatch_data.mem = numpy.zeros(
            (self.max_minibatch_size,) + self.shape, dtype=DTYPE)

        self.minibatch_labels.reset()
        if self.has_labels:
            self.minibatch_labels.mem = numpy.zeros(
                (self.max_minibatch_size,), dtype=Loader.LABEL_DTYPE)

        self.minibatch_indices.reset()
        self.minibatch_indices.mem = numpy.zeros(
            self.max_minibatch_size, dtype=Loader.INDEX_DTYPE)

    def keys_from_indices(self, indices):
        keys = []
        for si in indices:
            class_index, key_index = self.class_index_by_sample_index(si)
            keys.append(self.class_keys[class_index][key_index])
        return keys

    def fill_minibatch(self):
        keys = self.keys_from_indices(
            self.minibatch_indices.mem[:self.minibatch_size])
        assert self.has_labels == self.load_keys(
            keys, None, self.minibatch_data.mem, self.minibatch_labels.mem)


class ImageLoaderMSEMixin(LoaderMSEMixin):
    """
    Implementation of ImageLoaderMSE for parallel inheritance.

    Attributes:
        target_keys: additional key list of targets.
    """
    def __init__(self, workflow, **kwargs):
        super(ImageLoaderMSEMixin, self).__init__(workflow, **kwargs)
        self.target_keys = []
        self.target_label_map = None

    def load_data(self):
        super(ImageLoaderMSEMixin, self).load_data()
        if self._restored_from_pickle:
            return
        self.target_keys.extend(self.get_keys(TARGET))
        length = len(self.target_keys)
        if len(set(self.target_keys)) < length:
            raise error.BadFormatError("Some targets have duplicate keys")
        self.target_keys.sort()
        if not self.has_labels and length != sum(self.class_lengths):
            raise error.BadFormatError(
                "Number of class samples %d differs from the number of "
                "targets %d" % (sum(self.class_lengths), length))
        if self.has_labels:
            labels = numpy.zeros(length, dtype=Loader.LABEL_DTYPE)
            assert self.load_keys(self.target_keys, None, None, labels)
            if len(set(labels)) < length:
                raise error.BadFormatError("Targets have duplicate labels")
            self.target_label_map = {l: self.target_keys[l] for l in labels}

    def create_minibatches(self):
        super(ImageLoaderMSEMixin, self).create_minibatches()
        self.minibatch_targets.reset()
        self.minibatch_targets.mem = numpy.zeros(
            (self.max_minibatch_size,) + self.shape, dtype=DTYPE)

    def fill_minibatch(self):
        super(ImageLoaderMSEMixin, self).fill_minibatch()
        if not self.has_labels:
            keys = self.keys_from_indices(
                self.shuffled_indices[i]
                for i in self.minibatch_indices.mem[:self.minibatch_size])
        else:
            keys = []
            for label in self.minibatch_labels.mem:
                keys.append(self.target_label_map[label])
        assert self.has_labels == self.load_keys(
            keys, None, self.minibatch_targets.mem, None)


class ImageLoaderMSE(ImageLoader, ImageLoaderMSEMixin):
    """
    Loads images in MSE schemes. Like ImageLoader, mostly useful for large
    datasets.
    """
    pass


@implementer(IFullBatchLoader)
class FullBatchImageLoader(ImageLoader, FullBatchLoader):
    """Loads all images into the memory.
    """

    @property
    def has_labels(self):
        return ImageLoader.has_labels.fget(self)

    def load_data(self):
        super(FullBatchImageLoader, self).load_data()

        # Allocate data
        overall = sum(self.class_lengths)
        self.info("Found %d samples of shape %s (%d TEST, %d VALIDATION, "
                  "%d TRAIN)", overall, self.shape, *self.class_lengths)
        required_mem = overall * numpy.prod(self.shape) * numpy.dtype(
            self.source_dtype).itemsize
        if virtual_memory().free < required_mem:
            gb = 1.0 / (1000 * 1000 * 1000)
            self.critical("Not enough memory (free %.3f Gb, required %.3f Gb)",
                          virtual_memory().free * gb, required_mem * gb)
            raise MemoryError("Not enough memory")
        # Real allocation will still happen during the second pass
        self.original_data.mem = data = numpy.zeros(
            (overall,) + self.shape, dtype=self.source_dtype)
        self.original_labels.mem = labels = numpy.zeros(
            overall, dtype=Loader.LABEL_DTYPE)

        # Second pass
        pbar = ProgressBar(term_width=50, maxval=overall,
                           widgets=["Loading %d images " % overall,
                                    Bar(), ' ', Percentage()],
                           log_level=logging.INFO, poll=0.5)
        pbar.start()
        offset = 0
        has_labels = []
        for keys in self.class_keys:
            if len(keys) == 0:
                continue
            has_labels.append(self.load_keys(
                keys, pbar, data[offset:], labels[offset:]))
            offset += len(keys)
        pbar.finish()

        # Delete labels mem if no labels was extracted
        if numpy.prod(has_labels) == 0 and sum(has_labels) > 0:
            raise error.BadFormatError(
                "Some classes do not have labels while other do")
        if sum(has_labels) == 0:
            self.original_labels.mem = None


class FullBatchImageLoaderMSEMixin(ImageLoaderMSEMixin,
                                   FullBatchLoaderMSEMixin):
    """
    FullBatchImageLoaderMSE implementation for parallel inheritance.
    """

    def load_data(self):
        super(FullBatchImageLoaderMSEMixin, self).load_data()

        length = len(self.target_keys)
        targets = numpy.zeros((length,) + self.shape, dtype=self.source_dtype)
        target_labels = numpy.zeros(length, dtype=Loader.LABEL_DTYPE)
        has_labels = self.load_keys(
            self.target_keys, None, targets, target_labels)
        if not has_labels:
            if self.has_labels:
                raise error.BadFormatError(
                    "Targets do not have labels, but the classes do")
            # Associate targets with classes by sequence order
            self.original_targets.mem = targets
            return
        if not self.has_labels:
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


class FullBatchImageLoaderMSE(FullBatchImageLoader,
                              FullBatchImageLoaderMSEMixin):
    """
    MSE modification of FullBatchImageLoader class.
    """
    pass


class IFileImageLoader(Interface):
    def get_label_from_filename(filename):
        """Retrieves label for the specified file path.
        """

    def is_valid_filename(filename):
        """Filters the file names. Return True if the specified file path must
-        be included, otherwise, False.
        """


@implementer(IImageLoader)
class FileImageLoader(ImageLoader):
    """Loads images from multiple folders. As with ImageLoader, it is useful
    for large datasets.

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
        super(FileImageLoader, self).__init__(workflow, **kwargs)
        self.test_paths = kwargs.get("test_paths", [])
        self.validation_paths = kwargs.get("validation_paths", [])
        self.train_paths = kwargs.get("train_paths", [])
        self.verify_interface(IFileImageLoader)

    def _check_paths(self, paths):
        if not hasattr(paths, "__iter__"):
            raise TypeError("Paths must be iterable, e.g., a list or a tuple")

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
            with open(key, "rb") as fin:
                img = Image.open(fin)
                return (tuple(reversed(img.size)), MODE_COLOR_MAP[img.mode]),
        except Exception as e:
            self.warning("Failed to read %s with PIL: %s", key, e)
            # Unable to read the image with PIL. Fall back to slow OpenCV
            # method which reads the whole image.
            img = cv2.imread(key, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise error.BadFormatError("Unable to read %s" % key)
            return (img.shape[:2], "BGR"),

    def get_image_data(self, key):
        """
        Loads data from image and normalizes it.

        Returns:
            :class:`numpy.ndarrayarray`: if there was one image in the file.
            tuple: `(data, labels)` if there were many images in the file
        """
        try:
            with open(key, "rb") as fin:
                img = Image.open(fin)
                if img.mode in ("P", "CMYK"):
                    return numpy.array(img.convert("RGB"),
                                       dtype=self.source_dtype)
                else:
                    return numpy.array(img, dtype=self.source_dtype)
        except Exception as e:
            self.warning("Failed to read %s with PIL: %s", key, e)
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
            for size, color_space in infos:
                shape = size + (COLOR_CHANNELS_MAP[color_space],)
                if self.shape != tuple() and shape[:2] != self.shape[:2]:
                    self.warning("%s has the different shape %s (expected %s)",
                                 file, shape[:2], self.shape[:2])
                else:
                    if self.shape == tuple():
                        self.shape = shape
                    uniform_files.append(file)
        return uniform_files

    def get_keys(self, index):
        paths = (self.test_paths, self.validation_paths,
                 self.train_paths)[index]
        if paths is None:
            return []
        return list(chain.from_iterable(self.scan_files(p) for p in paths))


class FileImageLoaderMSEMixin(FullBatchImageLoaderMSEMixin):
    """
    FileImageLoaderMSE implementation for parallel inheritance.

    Attributes:
        target_paths: list of paths for target in case of MSE.
    """

    def __init__(self, workflow, **kwargs):
        super(FileImageLoaderMSEMixin, self).__init__(
            workflow, **kwargs)
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
            return super(FileImageLoaderMSEMixin, self).get_keys(
                index)
        return list(chain.from_iterable(
            self.scan_files(p) for p in self.target_paths))


class FileImageLoaderMSE(FileImageLoader, FileImageLoaderMSEMixin):
    """
    MSE modification of  FileImageLoader class.
    """
    pass


class FullBatchFileImageLoader(FileImageLoader, FullBatchImageLoader):
    """Loads images from multiple folders as full batch.
    """
    pass


class FullBatchFileImageLoaderMSE(FileImageLoaderMSEMixin,
                                  FullBatchImageLoaderMSEMixin):
    """
    MSE modification of  FullBatchFileImageLoader class.
    """
    pass
