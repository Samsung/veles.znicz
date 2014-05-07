"""
Created on Apr 10, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import unittest

from veles.tests import DummyWorkflow
from veles.znicz.samples.imagenet import LoaderDetection


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.loader = LoaderDetection(DummyWorkflow(),
                                      ipath="/data/imagenet/2013",
                                      dbpath="/data/imagenet/2013/db",
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
