"""
Created on Jan 23, 2015

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import bz2
import gzip
from io import SEEK_END
import lzma
import numpy
import os
from six import BytesIO
import snappy
from zope.interface import implementer
from veles import error
from veles.compat import from_none

from veles.config import root
from veles.pickle2 import pickle, best_protocol
from veles.snapshotter import Snapshotter
from veles.units import Unit, IUnit
from veles.znicz.loader import Loader, ILoader


@implementer(IUnit)
class MinibatchesSaver(Unit):
    """Saves data from Loader to pickle file.
    """
    def __init__(self, workflow, **kwargs):
        super(MinibatchesSaver, self).__init__(workflow, **kwargs)
        kwargs["view_group"] = kwargs.get("view_group", "SERVICE")
        self.file_name = kwargs.get(
            "file_name", os.path.join(root.common.cache_dir,
                                      "minibatches.sav"))
        self.file = None
        self.pickler = None
        self.compression = kwargs.get("compression", "snappy")
        self.compression_level = kwargs.get("compression_level", 9)
        self.offset_table = []
        self.demand(
            "minibatch_data", "minibatch_labels",
            "class_lengths", "max_minibatch_size", "shuffle_limit")

    def initialize(self, **kwargs):
        if self.shuffle_limit != 0:
            raise error.VelesException(
                "You must disable shuffling in your loader (set shuffle_limit "
                "to 0)")
        self.file = self.open_file()
        self.pickler = pickle.Pickler(self.file, protocol=best_protocol)
        try:
            self.pickler.fast = True
            self.debug("Activated FAST pickling mode")
        except AttributeError:
            pass
        pickle.dump(self.get_header_data(), self.file.fileobj,
                    protocol=best_protocol)

    def open_file(self):
        return Snapshotter.WRITE_CODECS[
            self.compression](self.file_name, self.compression_level)

    def get_header_data(self):
        return self.compression, self.class_lengths, self.max_minibatch_size, \
            self.minibatch_data.shape, self.minibatch_data.dtype, \
            self.minibatch_labels.shape, self.minibatch_labels.dtype

    def get_chunk_data(self):
        self.minibatch_data.map_read()
        self.minibatch_labels.map_read()
        return self.minibatch_data, self.minibatch_labels

    def run(self):
        self.offset_table.append(numpy.uint64(self.file.tell()))
        self.pickler.dump(self.get_chunk_data())

    def stop(self):
        self.file.flush()
        pickle.dump(self.offset_table, self.file.fileobj,
                    protocol=best_protocol)
        self.file.close()


def decompress_snappy(data):
    bio_in = BytesIO(data)
    bio_out = BytesIO()
    snappy.stream_decompress(bio_in, bio_out)
    return bio_out.getbuffer()


@implementer(ILoader)
class MinibatchesLoader(Loader):

    CODECS = {
        ".pickle": lambda b: b,
        "snappy": decompress_snappy,
        ".gz": gzip.decompress,
        ".bz2": bz2.decompress,
        ".xz": lzma.decompress,
    }

    def __init__(self, workflow, **kwargs):
        super(MinibatchesLoader, self).__init__(workflow, **kwargs)
        self.file_name = kwargs["file_name"]
        self.file = None
        self.offset_table = []
        self.minibatch_data_shape = None
        self.minibatch_data_dtype = None
        self.minibatch_labels_shape = None
        self.minibatch_labels_dtype = None
        self.decompress = None

    def load_data(self):
        self.file = open(self.file_name, "rb")
        (codec, self.class_lengths, self.max_minibatch_size,
         self.minibatch_data_shape, self.minibatch_data_dtype,
         self.minibatch_labels_shape, self.minibatch_labels_dtype) = \
            pickle.load(self.file)
        self.decompress = MinibatchesLoader.CODECS[codec]
        minibatches_count = sum(int(numpy.ceil(l / self.max_minibatch_size))
                                for l in self.class_lengths)

        class BytesMeasurer(object):
            def __init__(self):
                self.size = 0

            def write(self, data):
                self.size += len(data)

        bm = BytesMeasurer()
        fake_table = [numpy.uint64(i) for i in range(minibatches_count)]
        pickle.dump(fake_table, bm, protocol=best_protocol)
        self.file.seek(-bm.size, SEEK_END)
        try:
            self.offset_table = pickle.load(self.file)
        except pickle.UnpicklingError as e:
            self.error("Failed to read the offset table (table offset was %d)",
                       bm.size)
            raise from_none(e)
        # Virtual end
        self.offset_table.append(self.file.tell() - bm.size)

    def create_minibatches(self):
        self.minibatch_data.reset()
        self.minibatch_data.mem = numpy.zeros(
            self.minibatch_data_shape, dtype=self.minibatch_data_dtype)
        self.minibatch_labels.reset()
        self.minibatch_labels.mem = numpy.zeros(
            self.minibatch_labels_shape, dtype=self.minibatch_labels_dtype)
        self.minibatch_indices.reset()
        self.minibatch_indices.mem = numpy.zeros(
            self.max_minibatch_size, dtype=Loader.INDEX_DTYPE)

    def fill_minibatch(self):
        chunks_map = [self.get_address(i) + (i,) for i in
                      self.minibatch_indices.mem[:self.minibatch_size]]
        chunks_map.sort()
        prev_chunk_number = -1
        chunk = None
        for chunk_number, chunk_offset, index in chunks_map:
            if prev_chunk_number != chunk_number:
                prev_chunk_number = chunk_number
                self.file.seek(self.offset_table[chunk_number])
                buffer = self.file.read(self.offset_table[chunk_number + 1] -
                                        self.offset_table[chunk_number])
                chunk = pickle.loads(self.decompress(buffer))
            mb_data, mb_labels = chunk
            self.minibatch_data[index] = mb_data[chunk_offset]
            self.minibatch_labels[index] = mb_labels[chunk_offset]

    def get_address(self, index):
        class_index, class_offset = self.class_index_by_sample_index(index)
        chunk_number = sum(int(numpy.ceil(l / self.max_minibatch_size))
                           for i, l in enumerate(self.class_lengths)
                           if i < class_index)
        mb_ind, mb_off = divmod(class_offset, self.max_minibatch_size)
        chunk_number += mb_ind
        return chunk_number, mb_off
