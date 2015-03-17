"""
Created on Dec 9, 2014

A loader for LMDB base (CAFFE intermediate format).

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""

import numpy
from zope.interface import implementer

import lmdb
from veles.loader import IImageLoader, ImageLoader, CLASS_NAME
from veles.znicz.loader.caffe_pb2 import Datum


@implementer(IImageLoader)
class LMDBLoader(ImageLoader):
    MAPPING = "lmdb"

    def __init__(self, workflow, **kwargs):
        super(LMDBLoader, self).__init__(workflow, **kwargs)

        self._files = (kwargs.get("test_path", None),
                       kwargs.get("validation_path", None),
                       kwargs.get("train_path", None))

        self.original_shape = kwargs.get("db_shape", (256, 256, 3))
        self.db_color_space = kwargs.get("db_colorspace", "RGB")
        self.db_splitted_channels = kwargs.get("db_splitted_channels", True)

    def init_unpickled(self):
        super(LMDBLoader, self).init_unpickled()
        # LMDB base cursors, used as KV-iterators
        self._cursors_ = [None] * 3

    @property
    def files(self):
        return self._files

    @property
    def db_color_space(self):
        return self._db_color_space

    @db_color_space.setter
    def db_color_space(self, value):
        self._validate_color_space(value)
        self._db_color_space = value

    @property
    def db_splitted_channels(self):
        return self._db_splitted_channels

    @db_splitted_channels.setter
    def db_splitted_channels(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "db_splitted_channels must be boolean (got %s)" % type(value))
        self._db_splitted_channels = value

    def get_image_label(self, key):
        """Retrieves label for the specified key.
        """
        index, key = key
        datum = Datum()
        datum.ParseFromString(self._cursors_[index].get(key))
        return datum.label

    def get_image_info(self, key):
        """
        Return a tuple (size, color space).
        Size must be in OpenCV order (first y, then x),
        color space must be supported by OpenCV (COLOR_*).
        """
        index, key = key
        datum = Datum()
        datum.ParseFromString(self._cursors_[index].get(key))
        return (datum.height, datum.width), self.db_color_space

    def get_image_data(self, key):
        """Return the image data associated with the specified key.
        """
        index, key = key
        datum = Datum()
        datum.ParseFromString(self._cursors_[index].get(key))
        img = numpy.fromstring(datum.data, dtype=numpy.uint8)
        img = img.reshape(self.original_shape)
        if self.db_splitted_channels:
            img = img.swapaxes(0, 1).swapaxes(1, 2)
        return img

    def get_keys(self, index):
        """
        Return a list of image keys for the specified class index.
        """
        self._initialize_cursor(index)
        cursor = self._cursors_[index]
        keys = [(index, cursor.key())]
        while cursor.next():
            keys.append((index, cursor.key()))
        cursor.first()

        return keys

    def load_data(self):
        for index, _ in enumerate(CLASS_NAME):
            self._initialize_cursor(index)
        super(LMDBLoader, self).load_data()

    def _initialize_cursor(self, index):
        db_path = self._files[index]
        if not db_path:
            return tuple()

        _, cursor = self._open_db(db_path)
        self._cursors_[index] = cursor

    def _open_db(self, base_path):
        """
        Returns:
            int: number of pics in the database
            :class:`lmdb.Cursor`: base cursor
        """
        db = lmdb.open(base_path)
        transaction = db.begin()
        cursor = transaction.cursor()
        cursor.first()
        return db.stat()["entries"], cursor
