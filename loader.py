"""
Created on Aug 14, 2013

Loader base class.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import glob
import numpy
import scipy.ndimage
import time

import config
import error
import formats
import opencl_types
import rnd
import units


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
    def __init__(self, workflow, **kwargs):
        minibatch_max_size = kwargs.get("minibatch_max_size", 100)
        rnd_ = kwargs.get("rnd", rnd.default)
        kwargs["minibatch_max_size"] = minibatch_max_size
        kwargs["rnd"] = rnd_
        kwargs["view_group"] = kwargs.get("view_group", "LOADER")
        super(Loader, self).__init__(workflow, **kwargs)

        self.rnd = [rnd_]

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
            class_samples[].
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

        total_samples = 0
        for i, n in enumerate(self.class_samples):
            total_samples += n
            self.nextclass_offs[i] = total_samples
        self.total_samples[0] = total_samples
        if total_samples == 0:
            raise error.ErrBadFormat("class_samples should be filled in "
                                     "load_data()")

        # Adjust minibatch_maxsize.
        self.minibatch_maxsize[0] = min(self.minibatch_maxsize[0],
            max(self.class_samples[2], self.class_samples[1],
                self.class_samples[0]))

        self.create_minibatches()
        if self.minibatch_data.v == None:
            raise error.ErrBadFormat("minibatch_data MUST be initialized in "
                                     "create_minibatches()")

        self.minibatch_offs[0] = self.total_samples[0]

        # Initial shuffle.
        if self.shuffled_indexes == None:
            self.shuffled_indexes = numpy.arange(self.total_samples[0],
                dtype=opencl_types.itypes[
                    opencl_types.get_itype_from_size(self.total_samples[0])])

        if self.class_samples[0]:
            self.shuffle_validation_train()
        else:
            self.shuffle_train()

    def shuffle_validation_train(self):
        self.rnd[0].shuffle(self.shuffled_indexes[self.nextclass_offs[0]:
                                                  self.nextclass_offs[2]])

    def shuffle_train(self):
        self.rnd[0].shuffle(self.shuffled_indexes[self.nextclass_offs[1]:
                                                  self.nextclass_offs[2]])

    def shuffle(self):
        """Shuffle the dataset after one epoch.
        """
        self.shuffle_train()

    def fill_minibatch(self):
        """Fill minibatch data labels and indexes according to current shuffle.
        """
        pass

    def get_next_minibatch(self, minibatch_size):
        self.minibatch_offs[0] += minibatch_size
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

    def run(self):
        """Prepare the minibatch.
        """
        t1 = time.time()

        self.get_next_minibatch(self.minibatch_maxsize[0])
        minibatch_size = self.minibatch_size[0]

        # Fill minibatch according to current random shuffle and offset.
        self.minibatch_data.map_invalidate()
        self.minibatch_target.map_invalidate()
        self.minibatch_labels.map_invalidate()
        self.minibatch_indexes.map_invalidate()
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

        self.debug("%s in %.2f sec" % (self.__class__.__name__,
                                      time.time() - t1))

    def generate_data_for_slave(self, slave=None):
        self.get_next_minibatch(self.minibatch_maxsize[0])
        idxs = self.shuffled_indexes[self.minibatch_offs[0]:
                                     self.minibatch_offs[0] +
                                     self.minibatch_size[0]].copy()
        cls = self.minibatch_class[0]

        if not self.minibatch_last[0]:
            self.workflow.unlock_pipeline()

        return (idxs, cls)

    def apply_data_from_master(self, data):
        # Just feed single minibatch
        idxs = data[0]
        cls = data[1]
        minibatch_size = len(idxs)
        if minibatch_size > self.minibatch_maxsize[0]:
            raise error.ErrBadFormat("Received too many indexes from master")
        self.shuffled_indexes[:minibatch_size] = idxs[:]
        self.minibatch_size[0] = minibatch_size
        self.minibatch_offs[0] = -minibatch_size  # will be incremented in run
        self.total_samples[0] = minibatch_size
        self.minibatch_class[0] = cls
        self.minibatch_last[0] = 1
        for i in range(cls):
            self.nextclass_offs[i] = 0
        for i in range(cls, len(self.nextclass_offs)):
            self.nextclass_offs[i] = minibatch_size


class FullBatchLoader(Loader):
    """Loads data entire in memory.

    Attributes:
        original_data: numpy array of original data.
        original_labels: numpy array of original labels
                         (in case of classification).
        original_target: numpy array of original target
                         (in case of MSE).
        label_dtype: numpy dtype for label.

    Should be overriden in child class:
        load_data()
    """
    def __init__(self, workflow, **kwargs):
        super(FullBatchLoader, self).__init__(workflow, **kwargs)
        self.label_dtype = None

    def init_unpickled(self):
        super(FullBatchLoader, self).init_unpickled()
        self.original_data = None
        self.original_labels = None
        self.original_target = None
        self.shuffled_indexes = None

    def __getstate__(self):
        state = super(FullBatchLoader, self).__getstate__()
        state["original_data"] = None
        state["original_labels"] = None
        state["original_target"] = None
        state["shuffled_indexes"] = None
        return state

    def create_minibatches(self):
        if type(self.original_labels) == list:
            self.label_dtype = opencl_types.itypes[
                opencl_types.get_itype_from_size(
                    numpy.max(self.original_labels))]
        else:
            self.label_dtype = self.original_labels.dtype

        self.minibatch_data.reset()
        sh = [self.minibatch_maxsize[0]]
        sh.extend(self.original_data[0].shape)
        self.minibatch_data.v = numpy.zeros(sh,
                dtype=opencl_types.dtypes[config.c_dtype])

        self.minibatch_target.reset()
        if self.original_target != None:
            sh = [self.minibatch_maxsize[0]]
            sh.extend(self.original_target[0].shape)
            self.minibatch_target.v = numpy.zeros(sh,
                dtype=opencl_types.dtypes[config.c_dtype])

        self.minibatch_labels.reset()
        if self.original_labels != None:
            sh = [self.minibatch_maxsize[0]]
            self.minibatch_labels.v = numpy.zeros(sh,
                dtype=self.label_dtype)

        self.minibatch_indexes.reset()
        self.minibatch_indexes.v = numpy.zeros(len(self.original_data),
            dtype=opencl_types.itypes[opencl_types.get_itype_from_size(
                                len(self.original_data))])

    def fill_minibatch(self):
        super(FullBatchLoader, self).fill_minibatch()

        minibatch_size = self.minibatch_size[0]

        idxs = self.minibatch_indexes.v
        idxs[:minibatch_size] = self.shuffled_indexes[self.minibatch_offs[0]:
            self.minibatch_offs[0] + minibatch_size]

        for i, ii in enumerate(idxs[:minibatch_size]):
            self.minibatch_data.v[i] = self.original_data[ii]

        if self.original_labels != None:
            for i, ii in enumerate(idxs[:minibatch_size]):
                self.minibatch_labels.v[i] = self.original_labels[ii]

        if self.original_target != None:
            for i, ii in enumerate(idxs[:minibatch_size]):
                self.minibatch_target.v[i] = self.original_target[ii]

    def extract_validation_from_train(self, amount=0.15, rand=None):
        """Extracts validation dataset from train dataset randomly.

        We will rearrange indexes only.

        Parameters:
            amount: how many samples move from train dataset
                    relative to the entire samples count for each class.
            rand: rnd.Rand(), if None - will use self.rnd.
        """
        if rand == None:
            rand = self.rnd[0]
        if amount <= 0:  # Dispose of validation set
            self.class_samples[2] += self.class_samples[1]
            self.class_samples[1] = 0
            if self.shuffled_indexes == None:
                total_samples = numpy.sum(self.class_samples)
                self.shuffled_indexes = numpy.arange(total_samples,
                    dtype=opencl_types.itypes[
                        opencl_types.get_itype_from_size(total_samples)])
            return
        offs0 = self.class_samples[0]
        offs = offs0
        train_samples = self.class_samples[1] + self.class_samples[2]
        total_samples = train_samples + offs
        original_labels = self.original_labels

        if self.shuffled_indexes == None:
            self.shuffled_indexes = numpy.arange(total_samples,
                dtype=opencl_types.itypes[
                    opencl_types.get_itype_from_size(total_samples)])
        shuffled_indexes = self.shuffled_indexes

        # If there are no labels
        if original_labels == None:
            n = int(numpy.round(amount * train_samples))
            while n > 0:
                i = rand.randint(offs, offs + train_samples)

                # Swap indexes
                ii = shuffled_indexes[offs]
                shuffled_indexes[offs] = shuffled_indexes[i]
                shuffled_indexes[i] = ii

                offs += 1
                n -= 1
            self.class_samples[1] = offs - offs0
            self.class_samples[2] = (total_samples - self.class_samples[1] -
                                     offs0)
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
                raise error.ErrNotExists("There are too few labels "
                                         "for class %d" % (l))
            n += nn[l]
        while n > 0:
            i = rand.randint(offs, offs0 + train_samples)
            l = original_labels[shuffled_indexes[i]]
            if nn[l] <= 0:
                # Move unused label to the end

                # Swap indexes
                ii = shuffled_indexes[offs0 + train_samples - 1]
                shuffled_indexes[
                    offs0 + train_samples - 1] = shuffled_indexes[i]
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
        self.class_samples[1] = offs - offs0
        self.class_samples[2] = (total_samples - self.class_samples[1] - offs0)


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
        a = scipy.ndimage.imread(fnme, flatten=self.grayscale)
        a = a.astype(numpy.float32)
        formats.normalize(a)
        return a

    def get_label_from_filename(self, filename):
        """Returns label from filename.
        """
        pass

    def load_original(self, pathname):
        """Loads data from original files.
        """
        self.info("Loading from %s..." % (pathname))
        files = glob.glob(pathname)
        files.sort()
        n_files = len(files)
        if not n_files:
            self.warning("No files fetched as %s" % (pathname))
            return

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
                    raise error.ErrBadFormat("Found file with different "
                                             "size than first: %s", files[i])
                else:
                    sz = a.size
                lbl = self.get_label_from_filename(files[i])
                if lbl != None:
                    if type(lbl) != int:
                        raise error.ErrBadFormat("Found non-integer label "
                            "with type %s for %s" % (str(type(ll)), files[i]))
                    ll.append(lbl)
                if aa == None:
                    sh = [n_files]
                    sh.extend(a.shape)
                    aa = numpy.zeros(sh, dtype=a.dtype)
                next_samples = this_samples + 1
            else:
                a, l = obj[0], obj[1]
                if len(a) != len(l):
                    raise error.ErrBadFormat("from_image() returned different "
                                             "number of samples and labels.")
                if sz != -1 and a[0].size != sz:
                    raise error.ErrBadFormat("Found file with different sample"
                                             " size than first: %s", files[i])
                else:
                    sz = a[0].size
                ll.extend(l)
                if aa == None:
                    sh = [n_files + len(l) - 1]
                    sh.extend(a[0].shape)
                    aa = numpy.zeros(sh, dtype=a[0].dtype)
                next_samples = this_samples + len(l)
            if aa.shape[0] < next_samples:
                aa = numpy.append(aa, a, axis=0)
            aa[this_samples:next_samples] = a
            self.total_samples[0] += next_samples - this_samples
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

        if len(labels):
            max_ll = max(labels)
            self.info("Labels are indexed from-to: %d %d" % (
                            min(labels), max_ll))
            self.original_labels = numpy.array(labels,
                dtype=opencl_types.itypes[opencl_types.get_itype_from_size(max_ll)])

        # Loading target data and labels.
        if self.target_paths != None:
            n = 0
            for pathname in self.target_paths:
                (aa, ll) = self.load_original(pathname)
                if len(ll):  # there are labels
                    for i, label in enumerate(ll):
                        self.target_by_lbl[label] = aa[i]
                else:  # assume that target order is the same as data
                    for a in aa:
                        self.target_by_lbl[n] = a
                        n += 1
            if n:
                if n != numpy.sum(self.class_samples):
                    raise error.ErrBadFormat("Target samples count differs "
                                             "from data samples count.")
                self.original_labels = numpy.arange(n,
                    dtype=opencl_types.itypes[opencl_types.get_itype_from_size(n)])

        self.original_data = data

        target = None
        for aa in self.target_by_lbl.values():
            sh = [len(self.original_data)]
            sh.extend(aa.shape)
            target = numpy.zeros(sh, dtype=aa.dtype)
            break
        if target != None:
            for i, label in enumerate(self.original_labels):
                target[i] = self.target_by_lbl[label]
            self.target_by_lbl.clear()
        self.original_target = target
