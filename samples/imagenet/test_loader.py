"""
Created on Apr 10, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import unittest

from veles.tests import DummyWorkflow
from veles.znicz.samples.imagenet import Loader


class Test(unittest.TestCase):

    def testLoader(self):
        loader = Loader(DummyWorkflow(),
                        "/data/imagenet/2013", "/data/imagenet/2013/db",
                        "2013", "DET")
        loader.setup(level=logging.DEBUG)
        loader.load_data()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testLoader']
    unittest.main()
