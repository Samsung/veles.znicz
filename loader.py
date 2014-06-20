"""
Created on Aug 14, 2013

Loader base class.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division
from copy import copy
import numpy
import os
import time
from zope.interface import implementer, Interface

import veles.config as config
from veles.distributable import IDistributable
import veles.error as error
import veles.formats as formats
from veles.mutable import Bool
import veles.opencl_types as opencl_types
import veles.random_generator as random_generator
from veles.units import Unit, IUnit


TRAIN = 2
VALID = 1
TEST = 0
TRIAGE = {"train": TRAIN,
          "validation": VALID,
          "valid": VALID,
          "test": TEST}
CLASS_NAME = ["test", "validation", "train"]


class LoaderError(Exception):
    pass


class ILoader(Interface):
    def load_data():
        """Load the data here.

        Should be filled here:
            class_lengths[].
        """

    def create_minibatches():
        """Allocate arrays for minibatch_data etc. here.
        """

    def fill_minibatch():
        """Fill minibatch data labels and indexes according to current shuffle.
        """


@implementer(IUnit, IDistributable)
class Loader(Unit):
    """Loads data and provides minibatch output interface.

    Attributes:
        prng: veles.random_generator.RandomGenerator instance.
        normalize: normalize pixel values into [-1, 1] range.
                   True by default.
        validation_ratio: used by extract_validation_from_train() as a default
                          ratio.
        max_minibatch_size: maximal size of a minibatch.
        total_samples: total number of samples in the dataset.
        class_lengths: number of samples per class.
        class_offsets: offset in samples where the next class begins.
        class_targets: target for each class (in case of MSE).
        last_minibatch: if current minibatch is last in it's class.
        epoch_ended: True right after validation is completed and no samples
                     have been served since.
        epoch_number: current epoch number. Epoch ends when all validation set
                      is processed. If validation set is empty, it ends
                      after all training set is processed.
        minibatch_data: data (should be scaled usually scaled to [-1, 1]).
        minibatch_indices: global indices of images in minibatch.
        minibatch_labels: labels for indexes in minibatch (classification).
        minibatch_targets: target data (in case of MSE).
        shuffled_indices: indices for all dataset, shuffled with prng.
        samples_served: the total number of samples processed for all epochs.
        minibatch_class: current minibatch class.
        minibatch_offset: current minibatch offset.
        global_offset: first sample index which was not served during the
                       current epoch.
        minibatch_size: current minibatch size <= max_minibatch_size.
    """

    def __init__(self, workflow, **kwargs):
        kwargs["view_group"] = "LOADER"
        self.last_minibatch = Bool(False)
        super(Loader, self).__init__(workflow, **kwargs)
        self.verify_interface(ILoader)

        self.prng = kwargs.get("prng", random_generator.get())
        self.normalize = kwargs.get("normalize", True)
        self.validation_ratio = kwargs.get("validation_ratio", 0.15)
        self._max_minibatch_size = kwargs.get("minibatch_size", 100)
        if self._max_minibatch_size < 1:
            raise ValueError("minibatch_size must be greater than zero")

        self._total_samples = 0
        self.class_lengths = [0, 0, 0]
        self.class_offsets = [0, 0, 0]
        self.class_targets = formats.Vector()

        self.epoch_ended = Bool(False)
        self.epoch_number = 0

        self.samples_served = 0
        self.global_offset = 0

        self.minibatch_class = 0
        self.minibatch_data = formats.Vector()
        self.minibatch_targets = formats.Vector()
        self.minibatch_indices = formats.Vector()
        self.minibatch_labels = formats.Vector()

        self.failed_minibatches = {}
        self._total_failed = 0

    def init_unpickled(self):
        super(Loader, self).init_unpickled()
        self._minibatch_offset_ = 0
        self._minibatch_size_ = 0
        self.shuffled_indices = False
        self.pending_minibatches_ = {}
        self._minibatch_serve_timestamp_ = time.time()

    @property
    def shuffled_indices(self):
        return self._shuffled_indices_

    @shuffled_indices.setter
    def shuffled_indices(self, value):
        self._shuffled_indices_ = value

    @property
    def total_samples(self):
        return self._total_samples

    @total_samples.setter
    def total_samples(self, value):
        if value <= 0:
            raise error.BadFormatError("class_lengths should be filled")
        if value > numpy.iinfo(numpy.int32).max:
            raise NotImplementedError(
                "total_samples exceeds int32 capacity.")
        self._total_samples = value

    @property
    def samples_served(self):
        return self._samples_served

    @samples_served.setter
    def samples_served(self, value):
        self._samples_served = value
        if value == 0:
            return
        num, den = divmod(self.samples_served, self.total_samples)
        self.epoch_number = num
        if not self.is_slave:
            now = time.time()
            if now - self._minibatch_serve_timestamp_ >= 10:
                self._minibatch_serve_timestamp_ = now
                self.info("Served %d samples (%d epochs, %.1f%% current); "
                          "jobs failed: %d/pending: %d",
                          self.samples_served, num,
                          100. * den / self.total_samples,
                          len(self.failed_minibatches),
                          len(self.pending_minibatches_))

    @property
    def minibatch_class(self):
        return self._minibatch_class

    @minibatch_class.setter
    def minibatch_class(self, value):
        if not 0 <= value < len(CLASS_NAME):
            raise ValueError("Invalid minibatch_class value %s" % str(value))
        self._minibatch_class = value

    @property
    def minibatch_offset(self):
        return self._minibatch_offset_

    @minibatch_offset.setter
    def minibatch_offset(self, value):
        if not 0 <= value <= self.total_samples:
            raise ValueError("Invalid minibatch_offset value %s" % str(value))
        self._minibatch_offset_ = value
        self._update_flags()

    @property
    def minibatch_size(self):
        return self._minibatch_size_

    @minibatch_size.setter
    def minibatch_size(self, value):
        if not 0 < value <= self.max_minibatch_size:
            raise ValueError("Invalid minibatch_size value %s" % str(value))
        self._minibatch_size_ = value

    @property
    def max_minibatch_size(self):
        return self._max_minibatch_size

    @max_minibatch_size.setter
    def max_minibatch_size(self, value):
        if value < 1:
            raise ValueError("Invalid max_minibatch_size value %s" %
                             str(value))
        self._max_minibatch_size = min(value, max(self.class_lengths))
        if self._max_minibatch_size < 1:
            raise ValueError("max(self.class_lengths) is %d" %
                             max(self.class_lengths))
        self.info("Minibatch size is set to %d", self.max_minibatch_size)

    @property
    def minibatch_data(self):
        return self._minibatch_data

    @minibatch_data.setter
    def minibatch_data(self, value):
        self._minibatch_data = value

    @property
    def minibatch_targets(self):
        return self._minibatch_targets

    @minibatch_targets.setter
    def minibatch_targets(self, value):
        self._minibatch_targets = value

    @property
    def minibatch_indices(self):
        return self._minibatch_indices

    @minibatch_indices.setter
    def minibatch_indices(self, value):
        self._minibatch_indices = value

    @property
    def minibatch_labels(self):
        return self._minibatch_labels

    @minibatch_labels.setter
    def minibatch_labels(self, value):
        self._minibatch_labels = value

    @property
    def epoch_number(self):
        return self._epoch_number

    @epoch_number.setter
    def epoch_number(self, value):
        if value < 0:
            raise ValueError("epoch_number must be greater than or equal to 0")
        self._epoch_number = value

    @property
    def prng(self):
        """
        Returns the Pseudo Random Number Generator belonging to this instance.
        """
        return self._prng

    @prng.setter
    def prng(self, value):
        if not isinstance(value, random_generator.RandomGenerator):
            raise TypeError("prng must be an instance of RandomGenerator")
        self._prng = value

    @property
    def normalize(self):
        """
        True if sample values are normalized before being served; otherwise,
        False.
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        if not isinstance(value, bool):
            raise TypeError("normalize must be a boolean value")
        self._normalize = value

    @property
    def validation_ratio(self):
        return self._validation_ratio

    @validation_ratio.setter
    def validation_ratio(self, value):
        if isinstance(value, int):
            if value != 0:
                raise ValueError("validation_ratio must be in [0, 1).")
            self._validation_ratio = 0.0
            return
        if not isinstance(value, float):
            raise TypeError("validation_ratio must be a float")
        if value < 0 or value >= 1:
            raise ValueError("validation_ratio must be in [0, 1).")
        self._validation_ratio = value

    @property
    def class_ended(self):
        for offset in self.class_offsets:
            if self.global_offset == offset:
                return True
            if self.global_offset < offset:
                return False

    @property
    def total_failed(self):
        return self._total_failed

    def __getstate__(self):
        state = super(Loader, self).__getstate__()
        # Move all pending minibatches to failed set
        if not self.epoch_ended:
            state["failed_minibatches"] = copy(state["failed_minibatches"])
            state["failed_minibatches"].update(self.pending_minibatches_)
        else:
            state["failed_minibatches"] = {}
        return state

    def initialize(self, **kwargs):
        """Loads the data, initializes indices, shuffles the training set.
        """
        self.load_data()
        self._update_total_samples()
        self.info("Samples number: test: %d, validation: %d, train: %d",
                  *self.class_lengths)
        self.max_minibatch_size = kwargs.get("minibatch_size",
                                             self.max_minibatch_size)
        self.create_minibatches()
        if not self.minibatch_data:
            raise error.BadFormatError("minibatch_data MUST be initialized in "
                                       "create_minibatches()")
        self.shuffle()

    def run(self):
        """Prepares the minibatch.
        """
        if len(self.pending_minibatches_) > 0:
            self.pending_minibatches_.popitem()
        self._serve_next_minibatch(None)
        self._on_successful_serve()

    def generate_data_for_master(self):
        return True

    def generate_data_for_slave(self, slave):
        self._serve_next_minibatch(slave.id)
        data = {'indices': self.minibatch_indices.mem}
        for attr in ("minibatch_class", "minibatch_size", "minibatch_offset",
                     "epoch_number"):
            data[attr] = getattr(self, attr)
        data['last_minibatch'] = bool(self.last_minibatch)
        data['epoch_ended'] = bool(self.epoch_ended)
        self.has_data_for_slave = ((not self.class_ended) or
                                   len(self.failed_minibatches) > 0)
        return data

    def apply_data_from_master(self, data):
        # Just feed single minibatch
        for attr in ("minibatch_class", "minibatch_size", "minibatch_offset"):
            setattr(self, attr, data[attr])
        self.last_minibatch <<= data['last_minibatch']
        self.epoch_ended <<= data['epoch_ended']
        indices = data['indices']
        if len(indices) != self.minibatch_size:
            raise error.MasterSlaveCommunicationError(
                "minibatch size mismatch")
        if self.minibatch_offset > len(self.shuffled_indices):
            raise error.MasterSlaveCommunicationError(
                "minibatch offset overflow")
        if self.minibatch_offset - self.minibatch_size < 0:
            raise error.MasterSlaveCommunicationError(
                "minibatch offset - size < 0")
        # Patch shuffled_indices so that received indices will be picked up
        # during  _serve_next_minibatch()
        self.shuffled_indices[self.minibatch_offset - self.minibatch_size:
                              self.minibatch_offset] = indices

    def apply_data_from_slave(self, data, slave):
        if slave is None:
            # Partial update
            return
        try:
            self.minibatch_offset, self.minibatch_size = \
                self.pending_minibatches_[slave.id]
        except KeyError:
            raise error.Bug("pending_minibatches_ does not contain %s" %
                            slave.id)
        del self.pending_minibatches_[slave.id]
        self._on_successful_serve()
        if not self.has_data_for_slave:
            self.has_data_for_slave = self.last_minibatch

    def drop_slave(self, slave):
        if slave.id in self.pending_minibatches_:
            self._total_failed += 1
            self.failed_minibatches[slave.id] = \
                self.pending_minibatches_[slave.id]
            del self.pending_minibatches_[slave.id]
            self.has_data_for_slave = True
            self.info("Jobs failed: %d/pending: %d",
                      len(self.failed_minibatches),
                      len(self.pending_minibatches_))

    def extract_validation_from_train(self, rand=None, ratio=None):
        """Extracts validation dataset from train dataset randomly.

        We will rearrange indexes only.

        Parameters:
            amount: how many samples move from train dataset
                    relative to the entire samples count for each class.
            rand: veles.random_generator.RandomGenerator, if None - will use
                  self.prng.
        """
        amount = ratio or self.validation_ratio
        rand = rand or self.prng

        if amount <= 0:  # Dispose of validation set
            self.class_lengths[TRAIN] += self.class_lengths[VALID]
            self.class_lengths[VALID] = 0
            if self.shuffled_indices is False:
                total_samples = numpy.sum(self.class_lengths)
                self.shuffled_indices = numpy.arange(
                    total_samples, dtype=numpy.int32)
            return
        offs_test = self.class_lengths[TEST]
        offs = offs_test
        train_samples = self.class_lengths[VALID] + self.class_lengths[TRAIN]
        total_samples = train_samples + offs
        original_labels = self.original_labels

        if self.shuffled_indices is False:
            self.shuffled_indices = numpy.arange(
                total_samples, dtype=numpy.int32)
        shuffled_indexes = self.shuffled_indices

        # If there are no labels
        if original_labels is None:
            n = int(numpy.round(amount * train_samples))
            while n > 0:
                i = rand.randint(offs, offs + train_samples)

                # Swap indexes
                ii = shuffled_indexes[offs]
                shuffled_indexes[offs] = shuffled_indexes[i]
                shuffled_indexes[i] = ii

                offs += 1
                n -= 1
            self.class_lengths[VALID] = offs - offs_test
            self.class_lengths[TRAIN] = (total_samples
                                         - self.class_lengths[VALID]
                                         - offs_test)
            return
        # If there are labels
        nn = {}
        for i in shuffled_indexes[offs:]:
            l = original_labels[i]
            nn[l] = nn.get(l, 0) + 1
        n = 0
        for l in nn.keys():
            n_train = nn[l]
            nn[l] = max(int(numpy.round(amount * nn[l])), 1)
            if nn[l] >= n_train:
                raise error.NotExistsError("There are too few labels "
                                           "for class %d" % (l))
            n += nn[l]
        while n > 0:
            i = rand.randint(offs, offs_test + train_samples)
            l = original_labels[shuffled_indexes[i]]
            if nn[l] <= 0:
                # Move unused label to the end

                # Swap indexes
                ii = shuffled_indexes[offs_test + train_samples - 1]
                shuffled_indexes[
                    offs_test + train_samples - 1] = shuffled_indexes[i]
                shuffled_indexes[i] = ii

                train_samples -= 1
                continue
            # Swap indexes
            ii = shuffled_indexes[offs]
            shuffled_indexes[offs] = shuffled_indexes[i]
            shuffled_indexes[i] = ii

            nn[l] -= 1
            n -= 1
            offs += 1
        self.class_lengths[VALID] = offs - offs_test
        self.class_lengths[TRAIN] = (total_samples - self.class_lengths[VALID]
                                     - offs_test)

    def shuffle(self):
        """Randomly shuffles the TRAIN dataset.
        """
        if self.shuffled_indices is False:
            self.shuffled_indices = numpy.arange(self.total_samples,
                                                 dtype=numpy.int32)
        self.prng.shuffle(self.shuffled_indices[self.class_offsets[1]:
                                                self.class_offsets[2]])

    def _update_total_samples(self):
        """Fills self.class_offsets from self.class_lengths.
        """
        total_samples = 0
        for i, n in enumerate(self.class_lengths):
            total_samples += n
            self.class_offsets[i] = total_samples
        self.total_samples = total_samples
        if self.class_lengths[TRAIN] < 1:
            raise ValueError("class_length for TRAIN dataset is invalid: %d" %
                             self.class_lengths[TRAIN])

    def _serve_next_minibatch(self, slave):
        try:
            _, (minibatch_offset, minibatch_size) = \
                self.failed_minibatches.popitem()
        except KeyError:
            minibatch_offset, minibatch_size = self._advance_global_offset()
        self.pending_minibatches_[slave] = (minibatch_offset, minibatch_size)
        self.minibatch_offset, self.minibatch_size = \
            minibatch_offset, minibatch_size

        for v in (self.minibatch_data, self.minibatch_targets,
                  self.minibatch_labels, self.minibatch_indices):
            v.map_invalidate()
        self.minibatch_indices.mem = self.shuffled_indices[
            minibatch_offset - minibatch_size:minibatch_offset]
        self.fill_minibatch()
        if minibatch_size < self.max_minibatch_size:
            self.minibatch_data[minibatch_size:] = 0.0
            if self.minibatch_targets:
                self.minibatch_targets[minibatch_size:] = 0.0
            self.minibatch_labels[minibatch_size:] = -1
            self.minibatch_indices[minibatch_size:] = -1

    def _update_flags(self):
        """Resets epoch_ended and last_minibatch.
        """
        if self.is_slave:
            # The flags will be explicitly set in apply_data_from_master()
            return
        self.last_minibatch <<= (self.class_ended and
                                 len(self.pending_minibatches_) <= 1 and
                                 len(self.failed_minibatches) == 0)
        self.epoch_ended <<= self.last_minibatch and (
            self.minibatch_class == VALID or
            (self.minibatch_class == TRAIN and self.class_lengths[VALID] == 0))

    def _advance_global_offset(self):
        """Increments global_offset by an appropriate minibatch_size.
        """
        # Slave mode is much simpler than others
        if self.is_slave:
            return self.minibatch_offset, self.minibatch_size
        # Shuffle again when the end of data is reached.
        if self.global_offset >= self.total_samples:
            self.shuffle()
            self.global_offset = 0

        # Compute next minibatch class and size, updating epoch_ended and
        # last_minibatch
        for class_index, class_offset in enumerate(self.class_offsets):
            if self.global_offset < class_offset:
                self.minibatch_class = class_index
                remainder = class_offset - self.global_offset
                minibatch_size = min(remainder, self.max_minibatch_size)
                break
        else:
            raise error.Bug("Could not determine minibatch_class, "
                            "probably due to incorrect class_offsets.")

        self.global_offset += minibatch_size
        return self.global_offset, minibatch_size

    def _on_successful_serve(self):
        self.samples_served += self.minibatch_size
        if self.last_minibatch:
            self.info("Last minibatch (%d total) of class %s served",
                      self.class_lengths[self.minibatch_class],
                      CLASS_NAME[self.minibatch_class].upper())


class IFullBatchLoader(Interface):
    def load_data():
        """Load the data here.
        """


@implementer(ILoader)
class FullBatchLoader(Loader):
    """Loads data entire in memory.

    Attributes:
        original_data: numpy array of original data.
        original_labels: numpy array of original labels
                         (in case of classification).
        original_target: numpy array of original target
                         (in case of MSE).

    Should be overriden in child class:
        load_data()
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchLoader, self).__init__(workflow, **kwargs)
        self.verify_interface(IFullBatchLoader)

    def init_unpickled(self):
        super(FullBatchLoader, self).init_unpickled()
        self.original_data = False
        self.original_labels = False
        self.original_target = False
        self.shuffled_indices = False

    def __getstate__(self):
        state = super(FullBatchLoader, self).__getstate__()
        state["original_data"] = None
        state["original_labels"] = None
        state["original_target"] = None
        state["shuffled_indices"] = None
        return state

    def create_minibatches(self):
        self.minibatch_data.reset()
        sh = [self.max_minibatch_size]
        sh.extend(self.original_data[0].shape)
        self.minibatch_data.mem = numpy.zeros(
            sh, dtype=opencl_types.dtypes[config.root.common.precision_type])

        self.minibatch_targets.reset()
        if not self.original_target is False:
            sh = [self.max_minibatch_size]
            sh.extend(self.original_target[0].shape)
            self.minibatch_targets.mem = numpy.zeros(
                sh,
                dtype=opencl_types.dtypes[config.root.common.precision_type])

        self.minibatch_labels.reset()
        if not self.original_labels is False:
            sh = [self.max_minibatch_size]
            self.minibatch_labels.mem = numpy.zeros(sh, dtype=numpy.int32)

        self.minibatch_indices.reset()
        self.minibatch_indices.mem = numpy.zeros(self.max_minibatch_size,
                                                 dtype=numpy.int32)

    def fill_minibatch(self):
        idxs = self.minibatch_indices.mem

        for i, ii in enumerate(idxs[:self.minibatch_size]):
            self.minibatch_data[i] = self.original_data[int(ii)]

        if not self.original_labels is False:
            for i, ii in enumerate(idxs[:self.minibatch_size]):
                self.minibatch_labels[i] = self.original_labels[int(ii)]

        if not self.original_target is False:
            for i, ii in enumerate(idxs[:self.minibatch_size]):
                self.minibatch_targets[i] = self.original_target[int(ii)]


@implementer(IFullBatchLoader)
class ImageLoader(FullBatchLoader):
    """Loads images from multiple folders as full batch.

    Attributes:
        test_paths: list of paths with mask for test set,
                    for example: ["/tmp/*.png"].
        validation_paths: list of paths with mask for validation set,
                          for example: ["/tmp/*.png"].
        train_paths: list of paths with mask for train set,
                     for example: ["/tmp/*.png"].
        target_paths: list of paths for target in case of MSE.
        target_by_lbl: dictionary of targets by lbl
                       in case of classification and MSE.

    Should be overriden in child class:
        get_label_from_filename()
        is_valid_filename()
    """
    def __init__(self, workflow, **kwargs):
        test_paths = kwargs.get("test_paths")
        validation_paths = kwargs.get("validation_paths")
        train_paths = kwargs.get("train_paths")
        target_paths = kwargs.get("target_paths")
        grayscale = kwargs.get("grayscale", True)
        kwargs["test_paths"] = test_paths
        kwargs["validation_paths"] = validation_paths
        kwargs["train_paths"] = train_paths
        kwargs["target_paths"] = target_paths
        kwargs["grayscale"] = grayscale
        super(ImageLoader, self).__init__(workflow, **kwargs)
        self.test_paths = test_paths
        self.validation_paths = validation_paths
        self.train_paths = train_paths
        self.target_paths = target_paths
        self.grayscale = grayscale

    def init_unpickled(self):
        super(ImageLoader, self).init_unpickled()
        self.target_by_lbl = {}

    def from_image(self, fnme):
        """Loads data from image and normalizes it.

        Returns:
            numpy array: if there was one image in the file.
            tuple: (a, l) if there were many images in the file
                a - data
                l - labels.
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
            self.original_labels = numpy.array(labels, dtype=numpy.int32)

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
                self.original_labels = numpy.arange(n, dtype=numpy.int32)

        self.original_data = data

        target = False
        for aa in self.target_by_lbl.values():
            sh = [len(self.original_data)]
            sh.extend(aa.shape)
            target = numpy.zeros(sh, dtype=aa.dtype)
            break
        if target is not False:
            for i, label in enumerate(self.original_labels):
                target[i] = self.target_by_lbl[label]
            self.target_by_lbl.clear()
        self.original_target = target
