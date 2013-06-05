"""
Created on Mar 20, 2013

File for MNIST dataset.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import formats
import struct
import error
import numpy
import config


class MNISTLoader(units.Unit):
    """Loads MNIST data.

    State:
        output: contains MNIST images.
        labels: contains MNIST labels.
        from_set: contains info for each sample: 0-train, 1-validation, 2-test.
    """
    def __init__(self, unpickling=0):
        super(MNISTLoader, self).__init__(unpickling=unpickling)
        #self.test_only = False
        if unpickling:
            return
        self.output = formats.Batch()
        self.labels = formats.Labels(10)
        self.from_set = formats.Labels(3)

    def load_original(self, offs, labels_count, labels_fnme, images_fnme):
        """Loads data from original MNIST files.
        """
        print("One time relatively slow load from original MNIST files...")

        # Reading labels:
        fin = open(labels_fnme, "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2049:
            raise error.ErrBadFormat("Wrong header in train-labels")

        n_labels, = struct.unpack(">i", fin.read(4))
        if n_labels != labels_count:
            raise error.ErrBadFormat("Wrong number of labels in train-labels")

        arr = numpy.fromfile(fin, dtype=numpy.byte, count=n_labels)
        if arr.size != n_labels:
            raise error.ErrBadFormat("EOF reached while reading labels from "
                                     "train-labels")
        self.labels.batch[offs:offs + labels_count] = arr[:]
        if self.labels.batch.min() != 0 or self.labels.batch.max() != 9:
            raise error.ErrBadFormat("Wrong labels range in train-labels.")

        fin.close()

        # Reading images:
        fin = open(images_fnme, "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2051:
            raise error.ErrBadFormat("Wrong header in train-images")

        n_images, = struct.unpack(">i", fin.read(4))
        if n_images != n_labels:
            raise error.ErrBadFormat("Wrong number of images in train-images")

        n_rows, n_cols = struct.unpack(">2i", fin.read(8))
        if n_rows != 28 or n_cols != 28:
            raise error.ErrBadFormat("Wrong images size in train-images, "
                                     "should be 28*28")

        # 0 - white, 255 - black
        pixels = numpy.fromfile(fin, dtype=numpy.ubyte,
                                count=n_images * n_rows * n_cols)
        if pixels.shape[0] != n_images * n_rows * n_cols:
            raise error.ErrBadFormat("EOF reached while reading images "
                                     "from train-images")

        fin.close()

        # Transforming images into float arrays and normalizing to [-1, 1]:
        images = pixels.astype(config.dtypes[config.dtype]).\
            reshape(n_images, n_rows, n_cols)
        print("Original range: [%.1f, %.1f]" % (images.min(), images.max()))
        for image in images:
            vle_min = image.min()
            vle_max = image.max()
            image += -vle_min
            image *= 2.0 / (vle_max - vle_min)
            image += -1.0
        print("Range after normalization: [%.1f, %.1f]" % (images.min(),
                                                           images.max()))
        self.output.batch[offs:offs + n_images] = images[:]
        print("Done")

    def initialize(self):
        """Here we will load MNIST data.
        """
        if not self.labels.batch or self.labels.batch.size < 70000:
            self.labels.batch = numpy.zeros([70000], dtype=numpy.int8)
        if not self.from_set.batch or self.from_set.batch.size < 70000:
            self.from_set.batch = numpy.zeros([70000], dtype=numpy.int8)
        if not self.output.batch or self.output.batch.shape[0] < 70000:
            self.output.batch = numpy.zeros([70000, 28, 28],
                                            dtype=config.dtypes[config.dtype])

        self.load_original(0, 60000, "MNIST/train-labels.idx1-ubyte",
                           "MNIST/train-images.idx3-ubyte")
        self.from_set.batch[0:60000] = 0
        self.load_original(60000, 10000, "MNIST/t10k-labels.idx1-ubyte",
                           "MNIST/t10k-images.idx3-ubyte")
        self.from_set.batch[60000:70000] = 2

        self.output.update()
        self.labels.update()
        self.from_set.update()

    def run(self):
        """Just update an output.
        """
        self.output.update()
        self.labels.update()
        self.from_set.update()
