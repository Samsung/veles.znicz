"""
Created on Apr 10, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import unittest

from veles.tests import DummyWorkflow
from veles.znicz.samples.imagenet import Loader


loader = None


class Test(unittest.TestCase):
    def test_decode_image(self):
        data = loader._decode_image(0)
        self.assertEqual(3, len(data.shape))
        self.assertEqual(3, data.shape[2])
        self.assertGreater(data.shape[0], 100)
        self.assertGreater(data.shape[1], 100)

    def test_get_sample(self):
        data = loader._get_sample(450000)
        self.assertEqual(3, len(data.shape))
        self.assertEqual(3, data.shape[2])
        self.assertEqual(loader._data_shape[0], data.shape[0])
        self.assertEqual(loader._data_shape[1], data.shape[1])
        self.assertEqual(loader._dtype, data.dtype)
        loader._include_derivative = True
        loader._colorspace = "HSV"
        data = loader._get_sample(450000)
        self.assertEqual(3, len(data.shape))
        self.assertEqual(4, data.shape[2])
        self.assertEqual(loader._data_shape[0], data.shape[0])
        self.assertEqual(loader._data_shape[1], data.shape[1])
        self.assertEqual(loader._dtype, data.dtype)
        loader._include_derivative = False
        loader._colorspace = "RGB"

if __name__ == "__main__":
    loader = Loader(DummyWorkflow(),
                    ipath="/data/imagenet/2013",
                    dbpath="/data/imagenet/2013/db",
                    year="2013", series="DET")
    loader.setup(level=logging.DEBUG)
    loader.load_data()
    unittest.main()
