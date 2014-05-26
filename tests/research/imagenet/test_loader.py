"""
Created on Apr 10, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import os
import sys
import unittest

from veles.tests import DummyWorkflow
from veles.znicz.tests.research.imagenet import LoaderDetection


base_path = "/data/imagenet/2013"


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        lock_file = os.path.join(base_path, "db/LOCK")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except:
                print("Failed to remove", lock_file, file=sys.stderr)
                raise
        self.loader = LoaderDetection(DummyWorkflow(),
                                      ipath=base_path,
                                      dbpath=os.path.join(base_path, "db"),
                                      year="2013", series="DET")
        self.loader.setup(level=logging.DEBUG)
        self.loader.load_data()

    def test_decode_image(self):
        data = self.loader._decode_image(0)
        self.assertEqual(3, len(data.shape))
        self.assertEqual(3, data.shape[2])
        self.assertGreater(data.shape[0], 100)
        self.assertGreater(data.shape[1], 100)

    def test_get_sample(self):
        data = self.loader._get_sample(450000)
        self.assertEqual(3, len(data.shape))
        self.assertEqual(3, data.shape[2])
        self.assertEqual(self.loader._data_shape[0], data.shape[0])
        self.assertEqual(self.loader._data_shape[1], data.shape[1])
        self.assertEqual(self.loader._dtype, data.dtype)
        self.loader.include_derivative = True
        self.loader._colorspace = "HSV"
        data = self.loader._get_sample(450000)
        self.assertEqual(3, len(data.shape))
        self.assertEqual(4, data.shape[2])
        self.assertEqual(self.loader._data_shape[0], data.shape[0])
        self.assertEqual(self.loader._data_shape[1], data.shape[1])
        self.assertEqual(self.loader._dtype, data.dtype)
        self.loader.include_derivative = False
        self.loader._colorspace = "RGB"

if __name__ == "__main__":
    unittest.main()
