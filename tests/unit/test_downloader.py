"""
Created on Dec 4, 2013

Unit test for pooling layer forward propagation.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import os
import tarfile
import wget
import unittest

from veles.config import root
from veles.dummy import DummyWorkflow
from veles.znicz.downloader import Downloader


class TestDownloader(unittest.TestCase):
    def test(self):
        downloader = Downloader(
            DummyWorkflow(),
            url="",
            directory=root.common.cache_dir,
            files=["txt_file.txt"])
        txt_file = os.path.join(downloader.directory, "txt_file.txt")
        file = os.path.join(downloader.directory, "TestDownloader.tar")
        with open(txt_file, "w") as fout:
            fout.write("Some text")
        with tarfile.open(file, mode='w') as fout:
            fout.add(txt_file, arcname="txt_file.txt")
        os.remove(txt_file)
        wget.download = lambda url, d: file
        downloader.initialize()
        if not os.path.exists(os.path.join(downloader.directory,
                                           "txt_file.txt")):
            raise OSError(
                "File %s not found" %
                os.path.join(downloader.directory, "txt_file.txt"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
