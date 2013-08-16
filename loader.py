"""
Created on Aug 14, 2013

Loader base class.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import config
import rnd
import formats
import time
import error
import numpy


class Loader(units.Unit):
    """Loads data and provides mini-batch output interface.

    Attributes:
        rnd: rnd.Rand().

        minibatch_data: data (should be scaled usually scaled to [-1, 1]).
        minibatch_indexes: global indexes of images in minibatch.
        minibatch_labels: labels for indexes in minibatch
                          (in case of classification task).
        minibatch_target: target data (in case of MSE).
        class_target: target for each class
                      (in case of classification with MSE).

        minibatch_class: class of the minibatch: 0-test, 1-validation, 2-train.
        minibatch_last: if current minibatch is last in it's class.

        minibatch_offs: offset of the current minibatch in all samples,
                        where first come test samples, then validation, with
                        train ones at the end.
        minibatch_size: size of the current minibatch.
        minibatch_maxsize: maximum size of minibatch in samples.

        total_samples: total number of samples in the dataset.
        class_samples: number of samples per class.
        nextclass_offs: offset in samples where the next class begins.

        shuffled_indexes: indexes for all dataset, shuffled with rnd.

    Should be overriden in child class:
        load_data()
        create_minibatches()
        fill_minibatch()
    """
    def __init__(self, minibatch_max_size=100, rnd=rnd.default):
        super(Loader, self).__init__()

        self.rnd = [rnd]

        self.minibatch_data = formats.Vector()
        self.minibatch_target = formats.Vector()
        self.minibatch_indexes = formats.Vector()
        self.minibatch_labels = formats.Vector()
        self.class_target = formats.Vector()

        self.minibatch_class = [0]
        self.minibatch_last = [0]

        self.total_samples = [0]
        self.class_samples = [0, 0, 0]
        self.nextclass_offs = [0, 0, 0]

        self.minibatch_offs = [0]
        self.minibatch_size = [0]
        self.minibatch_maxsize = [minibatch_max_size]

        self.shuffled_indexes = None

    def __getstate__(self):
        state = super(Loader, self).__getstate__()
        state["shuffled_indexes"] = None
        return state

    def load_data(self):
        """Load the data here.

        Should be filled here:
            total_samples, class_samples, nextclass_offs.
        """
        pass

    def create_minibatches(self):
        """Allocate arrays for minibatch_data etc. here.
        """
        pass

    def initialize(self):
        res = self.load_data()
        if res:
            return res

        # Check for correctness.
        if self.total_samples[0] == 0:
            raise error.ErrBadFormat("class_samples, nextclass_offs "
                "and total_samples should be initialized after load_data(): "
                "got self.total_samples[0] == 0")
        if self.total_samples[0] != self.nextclass_offs[2]:
            raise error.ErrBadFormat("self.total_samples[0] != "
                "self.nextclass_offs[2] (%d != %d)" % (
                self.total_samples[0], self.nextclass_offs[2]))
        offs = 0
        for i in range(0, 3):
            offs += self.class_samples[i]
            if self.nextclass_offs[i] != offs:
                raise error.ErrBadFormat("self.nextclass_offs[%d] != "
                    "%d" % (self.nextclass_offs[i], offs))

        # Adjust minibatch_maxsize.
        self.minibatch_maxsize[0] = min(self.minibatch_maxsize[0],
            max(self.class_samples[2], self.class_samples[1],
                self.class_samples[0]))

        res = self.create_minibatches()
        if res:
            return res
        if self.minibatch_data.v == None:
            raise error.ErrBadFormat("minibatch_data MUST be initialized in "
                                     "create_minibatches()")

        self.minibatch_offs[0] = self.total_samples[0]

        # Initial shuffle.
        self.shuffled_indexes = numpy.arange(self.total_samples[0],
            dtype=config.itypes[
                config.get_itype_from_size(self.total_samples[0])])

        if self.class_samples[0]:
            self.shuffle_validation_train()
        else:
            self.shuffle_train()

    def shuffle_validation_train(self):
        self.rnd[0].shuffle(self.shuffled_indexes[self.nextclass_offs[0]:\
                                                  self.nextclass_offs[2]])

    def shuffle_train(self):
        self.rnd[0].shuffle(self.shuffled_indexes[self.nextclass_offs[1]:\
                                                  self.nextclass_offs[2]])

    def shuffle(self):
        """Shuffle the dataset after one epoch.
        """
        self.shuffle_train()

    def fill_minibatch(self):
        """Fill minibatch data labels and indexes according to current shuffle.
        """
        pass

    def run(self):
        """Prepare the minibatch.
        """
        t1 = time.time()

        self.minibatch_offs[0] += self.minibatch_size[0]
        # Reshuffle when end of data reached.
        if self.minibatch_offs[0] >= self.total_samples[0]:
            self.shuffle()
            self.minibatch_offs[0] = 0

        # Compute minibatch size and it's class.
        for i in range(0, len(self.nextclass_offs)):
            if self.minibatch_offs[0] < self.nextclass_offs[i]:
                self.minibatch_class[0] = i
                minibatch_size = min(self.minibatch_maxsize[0],
                    self.nextclass_offs[i] - self.minibatch_offs[0])
                if self.minibatch_offs[0] + minibatch_size >= \
                   self.nextclass_offs[self.minibatch_class[0]]:
                    self.minibatch_last[0] = 1
                else:
                    self.minibatch_last[0] = 0
                break
        else:
            raise error.ErrNotExists("Could not determine minibatch class.")
        self.minibatch_size[0] = minibatch_size

        # Fill minibatch according to current random shuffle and offset.
        self.fill_minibatch()

        # Fill excessive indexes.
        if minibatch_size < self.minibatch_maxsize[0]:
            self.minibatch_data.v[minibatch_size:] = 0.0
            if self.minibatch_target.v != None:
                self.minibatch_target.v[minibatch_size:] = 0.0
            if self.minibatch_labels != None:
                self.minibatch_labels.v[minibatch_size:] = -1
            if self.minibatch_indexes != None:
                self.minibatch_indexes.v[minibatch_size:] = -1

        # Set update flag for GPU operation.
        self.minibatch_data.update()
        self.minibatch_target.update()
        self.minibatch_indexes.update()
        self.minibatch_labels.update()

        self.log().debug("%s in %.2f sec" % (self.__class__.__name__,
                                      time.time() - t1))


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
    def init_unpickled(self):
        super(FullBatchLoader, self).init_unpickled()
        self.original_data = None
        self.original_labels = None
        self.original_target = None

    def __getstate__(self):
        state = super(FullBatchLoader, self).__getstate__()
        state["original_data"] = None
        state["original_labels"] = None
        state["original_target"] = None
        return state

    def create_minibatches(self):
        self.minibatch_data.reset()
        sh = [self.minibatch_maxsize[0]]
        sh.extend(self.original_data.shape[1:])
        self.minibatch_data.v = numpy.zeros(sh,
                dtype=self.original_data.dtype)

        self.minibatch_target.reset()
        if self.original_target != None:
            sh = [self.minibatch_maxsize[0]]
            sh.extend(self.original_target.shape[1:])
            self.minibatch_target.v = numpy.zeros(sh,
                dtype=self.original_target.dtype)

        self.minibatch_labels.reset()
        if self.original_labels != None:
            sh = [self.minibatch_maxsize[0]]
            self.minibatch_labels.v = numpy.zeros(sh,
                dtype=self.original_labels.dtype)

        self.minibatch_indexes.reset()
        self.minibatch_indexes.v = numpy.zeros(len(self.original_data),
            dtype=config.itypes[config.get_itype_from_size(
                                len(self.original_data))])

    def fill_minibatch(self):
        super(FullBatchLoader, self).fill_minibatch()

        minibatch_size = self.minibatch_size[0]

        idxs = self.minibatch_indexes.v
        idxs[0:minibatch_size] = self.shuffled_indexes[self.minibatch_offs[0]:\
            self.minibatch_offs[0] + minibatch_size]

        self.minibatch_data.v[0:minibatch_size] = \
            self.original_data[idxs[0:minibatch_size]]

        if self.original_labels != None:
            self.minibatch_labels.v[0:minibatch_size] = \
                self.original_labels[idxs[0:minibatch_size]]

        if self.original_target != None:
            self.minibatch_target.v[0:minibatch_size] = \
                self.original_target[idxs[0:minibatch_size]]


import glob
import scipy.ndimage


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
    """
    def __init__(self, minibatch_max_size=100,
                 test_paths=None, validation_paths=None, train_paths=None,
                 target_paths=None, grayscale=True, rnd=rnd.default):
        super(ImageLoader, self).__init__(
            minibatch_max_size=minibatch_max_size, rnd=rnd)
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

        Override to resize if neccessary.
        """
        a = scipy.ndimage.imread(fnme, flatten=self.grayscale)
        a = a.astype(config.dtypes[config.dtype])
        formats.normalize(a)
        return a

    def get_label_from_filename(self, filename):
        """Returns label from filename.
        """
        pass

    def load_original(self, pathname):
        """Loads data from original files.
        """
        self.log().info("Loading from %s..." % (pathname))
        files = glob.glob(pathname)
        files.sort()
        n_files = len(files)
        if not n_files:
            self.log().warning("No files fetched as %s" % (pathname))
            return

        aa = None
        ll = []

        sz = -1
        for i in range(0, n_files):
            a = self.from_image(files[i])
            if sz != -1 and a.size != sz:
                raise error.ErrBadFormat("Found file with different "
                                         "size than first: %s", files[i])
            else:
                sz = a.size
            if aa == None:
                sh = [n_files]
                sh.extend(a.shape)
                aa = numpy.zeros(sh, dtype=config.dtypes[config.dtype])
            aa[i] = a
            lbl = self.get_label_from_filename(files[i])
            if lbl != None:
                if type(lbl) != int:
                    raise error.ErrBadFormat("Found non-integer label "
                        "with type %s for %s" % (str(type(ll)), files[i]))
                ll.append(lbl)

        return (aa, ll)

    def load_data(self):
        data = None
        labels = []

        # Loading original data and labels.
        offs = 0
        i = -1
        for t in (self.test_paths, self.validation_paths, self.train_paths):
            i += 1
            if t == None or not len(t):
                continue
            for pathname in t:
                (aa, ll) = self.load_original(pathname)
                if not len(aa):
                    continue
                if len(ll):
                    if len(ll) != len(aa):
                        raise error.ErrBadFormat("Number of labels %d differs "
                            "from number of input images %d for %s" % (len(ll),
                                len(aa), pathname))
                    labels.extend(ll)
                elif len(labels):
                    raise error.ErrBadFormat("Not labels found for %s" % (
                                                                    pathname))
                if data == None:
                    data = aa
                else:
                    data = numpy.append(data, aa, axis=0)
            self.class_samples[i] = len(data) - offs
            offs = len(data)
            self.nextclass_offs[i] = offs

        self.total_samples[0] = len(data)
        if len(labels):
            max_ll = max(labels)
            self.log().info("Labels are indexed from-to: %d %d" % (
                            min(labels), max_ll))
            self.original_labels = numpy.array(labels,
                dtype=config.itypes[config.get_itype_from_size(max_ll)])

        # Loading target data and labels.
        if self.target_paths != None:
            for pathname in self.target_paths:
                (aa, ll) = self.load_original(pathname)
                for i, label in enumerate(ll):
                    self.target_by_lbl[label] = aa[i]

        self.original_data = data

        target = None
        for aa in self.target_by_lbl.values():
            sh = [len(self.original_data)]
            sh.extend(aa.shape)
            target = numpy.zeros(sh, dtype=config.dtypes[config.dtype])
            break
        if target != None:
            for i, label in enumerate(self.original_labels):
                target[i] = self.target_by_lbl[label]
                self.target_by_lbl.pop(label)
        self.original_target = target
