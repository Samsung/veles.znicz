#!/usr/bin/python3 -O
"""
Created on Dec 4, 2014

Wine Loader file.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
from zope.interface import implementer

from veles.config import root, get
import veles.formats as formats
import veles.znicz.loader as loader

data_path = os.path.abspath(get(
    root.wine.loader.base, os.path.dirname(__file__)))

root.wine.loader.dataset_file = os.path.join(data_path, "wine.txt.gz")


@implementer(loader.IFullBatchLoader)
class WineLoader(loader.FullBatchLoader):
    """Loads Wine dataset.
    """
    def load_data(self):
        arr = numpy.loadtxt(root.wine.loader.dataset_file, delimiter=',')
        self.original_data.mem = arr[:, 1:]
        self.original_labels.mem = arr[:, 0].ravel().astype(numpy.int32) - 1

        IMul, IAdd = formats.normalize_pointwise(self.original_data.mem)
        self.original_data.mem[:] *= IMul
        self.original_data.mem[:] += IAdd

        self.class_lengths[0] = 0
        self.class_lengths[1] = 0
        self.class_lengths[2] = self.original_data.shape[0]
