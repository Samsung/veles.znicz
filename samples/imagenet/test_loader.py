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
                        ipath="/data/imagenet/2013",
                        dbpath="/data/imagenet/2013/db",
                        year="2013", series="DET")
        loader.setup(level=logging.DEBUG)
        loader.load_data()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testLoader']
    unittest.main()
