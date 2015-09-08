# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Dec 4, 2014

Loads MNIST dataset files.

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


import gzip
import numpy
import os
import struct
import wget
from zope.interface import implementer

import veles.error as error
from veles.loader import FullBatchLoader, IFullBatchLoader, TEST, VALID, TRAIN


@implementer(IFullBatchLoader)
class MnistLoader(FullBatchLoader):
    """Loads MNIST dataset.
    """
    MAPPING = "mnist_loader"

    def __init__(self, workflow, **kwargs):
        super(MnistLoader, self).__init__(workflow, **kwargs)
        self.data_path = kwargs["data_path"]
        self.test_labels = os.path.join(
            self.data_path, "t10k-labels.idx1-ubyte")
        self.test_data = os.path.join(self.data_path, "t10k-images.idx3-ubyte")
        self.train_labels = os.path.join(
            self.data_path, "train-labels.idx1-ubyte")
        self.train_data = os.path.join(
            self.data_path, "train-images.idx3-ubyte")
        self.url = "http://yann.lecun.com/exdb/mnist"

    def load_dataset(self):
        """
        Loads dataset from internet
        """
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if (os.path.exists(self.test_data) and
                os.path.exists(self.test_labels) and
                os.path.exists(self.train_data) and
                os.path.exists(self.train_labels)):
            return

        self.warning("Some file in %s does not exist, downloading from %s...",
                     self.data_path, self.url)

        files = {"train-images-idx3-ubyte.gz": "train-images.idx3-ubyte",
                 "train-labels-idx1-ubyte.gz": "train-labels.idx1-ubyte",
                 "t10k-images-idx3-ubyte.gz": "t10k-images.idx3-ubyte",
                 "t10k-labels-idx1-ubyte.gz": "t10k-labels.idx1-ubyte"}

        os.mkdir(self.data_path)
        for index, (k, v) in enumerate(sorted(files.items())):
            self.info("%d/%d", index + 1, len(files))
            wget.download("%s/%s" % (self.url, k), self.data_path)
            print("")
            with open(os.path.join(self.data_path, v), "wb") as fout:
                gz_file = os.path.join(self.data_path, k)
                with gzip.GzipFile(gz_file) as fin:
                    fout.write(fin.read())
                os.remove(gz_file)

    def load_original(self, offs, labels_count, labels_fnme, images_fnme):
        """Loads data from original MNIST files.
        """
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
        if not self.testing:
            self.class_lengths[TEST] = 0
            self.class_lengths[VALID] = 10000
            self.class_lengths[TRAIN] = 60000
        else:
            self.class_lengths[TEST] = 70000
            self.class_lengths[VALID] = self.class_lengths[TRAIN] = 0
        self.create_originals((28, 28))
        self.original_labels[:] = (0 for _ in range(len(self.original_labels)))
        self.info("Loading from original MNIST files...")
        self.load_dataset()
        if not (os.path.exists(self.test_data) and
                os.path.exists(self.test_labels) and
                os.path.exists(self.train_data) and
                os.path.exists(self.train_labels)):
            raise OSError(
                "There are no data in %s. Failed to load data from url: %s. "
                "Please download MNIST dataset manualy in folder %s" %
                (self.data_path, self.url, self.data_path))
        self.load_original(0, 10000, self.test_labels, self.test_data)
        self.load_original(10000, 60000, self.train_labels, self.train_data)
