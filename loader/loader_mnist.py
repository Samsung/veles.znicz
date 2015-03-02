#!/usr/bin/python3 -O
"""
Created on Dec 4, 2014

Loader Mnist file.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import os
import struct

import numpy
from zope.interface import implementer

from veles.config import root
import veles.error as error
import veles.loader as loader


mnist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "MNIST"))
if not os.access(mnist_dir, os.W_OK):
    # Fall back to ~/.veles/MNIST
    mnist_dir = os.path.join(root.common.veles_user_dir, "MNIST")
test_image_dir = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
test_label_dir = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
train_image_dir = os.path.join(mnist_dir, "train-images.idx3-ubyte")
train_label_dir = os.path.join(mnist_dir, "train-labels.idx1-ubyte")


@implementer(loader.IFullBatchLoader)
class MnistLoader(loader.FullBatchLoader):
    """Loads MNIST dataset.
    """
    MAPPING = "mnist_loader"

    def load_original(self, offs, labels_count, labels_fnme, images_fnme):
        """Loads data from original MNIST files.
        """
        if not os.path.exists(mnist_dir):
            url = "http://yann.lecun.com/exdb/mnist"
            self.warning("%s does not exist, downloading from %s...",
                         mnist_dir, url)

            import gzip
            import wget

            files = {"train-images-idx3-ubyte.gz": "train-images.idx3-ubyte",
                     "train-labels-idx1-ubyte.gz": "train-labels.idx1-ubyte",
                     "t10k-images-idx3-ubyte.gz": "t10k-images.idx3-ubyte",
                     "t10k-labels-idx1-ubyte.gz": "t10k-labels.idx1-ubyte"}

            os.mkdir(mnist_dir)
            for index, (k, v) in enumerate(sorted(files.items())):
                self.info("%d/%d", index + 1, len(files))
                wget.download("%s/%s" % (url, k), mnist_dir)
                print("")
                with open(os.path.join(mnist_dir, v), "wb") as fout:
                    gz_file = os.path.join(mnist_dir, k)
                    with gzip.GzipFile(gz_file) as fin:
                        fout.write(fin.read())
                    os.remove(gz_file)

        # Reading labels:
        with open(labels_fnme, "rb") as fin:
            header, = struct.unpack(">i", fin.read(4))
            if header != 2049:
                raise error.BadFormatError("Wrong header in train-labels")

            n_labels, = struct.unpack(">i", fin.read(4))
            if n_labels != labels_count:
                raise error.BadFormatError("Wrong number of labels in "
                                           "train-labels")

            arr = numpy.zeros(n_labels, dtype=numpy.byte)
            n = fin.readinto(arr)
            if n != n_labels:
                raise error.BadFormatError("EOF reached while reading labels "
                                           "from train-labels")
            self.original_labels[offs:offs + labels_count] = arr[:]
            if (numpy.min(self.original_labels) != 0 or
                    numpy.max(self.original_labels) != 9):
                raise error.BadFormatError(
                    "Wrong labels range in train-labels.")

        # Reading images:
        with open(images_fnme, "rb") as fin:
            header, = struct.unpack(">i", fin.read(4))
            if header != 2051:
                raise error.BadFormatError("Wrong header in train-images")

            n_images, = struct.unpack(">i", fin.read(4))
            if n_images != n_labels:
                raise error.BadFormatError("Wrong number of images in "
                                           "train-images")

            n_rows, n_cols = struct.unpack(">2i", fin.read(8))
            if n_rows != 28 or n_cols != 28:
                raise error.BadFormatError("Wrong images size in train-images,"
                                           " should be 28*28")

            # 0 - white, 255 - black
            pixels = numpy.zeros(n_images * n_rows * n_cols, dtype=numpy.ubyte)
            n = fin.readinto(pixels)
            if n != n_images * n_rows * n_cols:
                raise error.BadFormatError("EOF reached while reading images "
                                           "from train-images")

        # Transforming images into float arrays and normalizing to [-1, 1]:
        images = pixels.astype(numpy.float32).reshape(n_images, n_rows, n_cols)
        self.original_data.mem[offs:offs + n_images] = images[:]

    def load_data(self):
        """Here we will load MNIST data.
        """
        self.class_lengths[0] = 0
        self.class_lengths[1] = 10000
        self.class_lengths[2] = 60000
        self.create_originals((28, 28))
        self.original_labels[:] = (0 for _ in range(len(self.original_labels)))
        self.info("Loading from original MNIST files...")
        self.load_original(0, 10000, test_label_dir, test_image_dir)
        self.load_original(10000, 60000, train_label_dir, train_image_dir)
