# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Dec 4, 2013

Unit test for pooling layer forward propagation.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
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
