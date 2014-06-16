"""
Created on Apr 10, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import os
from six import print_
import sys
import unittest

from veles.tests import DummyWorkflow
from veles.znicz.tests.research.imagenet import LoaderDetection


base_path = "/data/imagenet/2013"


class Test(unittest.TestCase):
    loader = None

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        lock_file = os.path.join(base_path, "db/LOCK")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except:
                print_("Failed to remove", lock_file, file=sys.stderr)
                raise
        if Test.loader is None:
            Test.loader = LoaderDetection(DummyWorkflow(),
                                          ipath=base_path,
                                          dbpath=os.path.join(base_path, "db"),
                                          year="2013", series="DET")
            Test.loader.setup(level=logging.DEBUG)
            Test.loader.load_data()

    def test_decode_image(self):
        data = Test.loader._decode_image(0)
        self.assertEqual(3, len(data.shape))
        self.assertEqual(3, data.shape[2])
        self.assertGreater(data.shape[0], 100)
        self.assertGreater(data.shape[1], 100)

    def test_get_sample(self):
        data = Test.loader._get_sample(450000)
        self.assertEqual(3, len(data.shape))
        self.assertEqual(3, data.shape[2])
        self.assertEqual(Test.loader._data_shape[0], data.shape[0])
        self.assertEqual(Test.loader._data_shape[1], data.shape[1])
        self.assertEqual(Test.loader._dtype, data.dtype)
        Test.loader.include_derivative = True
        Test.loader._colorspace = "HSV"
        data = Test.loader._get_sample(450000)
        self.assertEqual(3, len(data.shape))
        self.assertEqual(4, data.shape[2])
        self.assertEqual(Test.loader._data_shape[0], data.shape[0])
        self.assertEqual(Test.loader._data_shape[1], data.shape[1])
        self.assertEqual(Test.loader._dtype, data.dtype)
        Test.loader.include_derivative = False
        Test.loader._colorspace = "RGB"

if __name__ == "__main__":
    unittest.main()
