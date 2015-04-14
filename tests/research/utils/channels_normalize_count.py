#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Sep 2, 2013

Will normalize counts of *.jp2 in the supplied folder
by replicating some of the found files.

.. argparse::
   :module: veles.znicz.tests.research.imagenet.preparation_imagenet
   :func: create_args_parser_sphinx
   :prog: preparation_imagenet

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
"""
Created on Sep 2, 2013

Will normalize counts of *.jp2 in the supplied folder
by replicating some of the found files.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import argparse
import numpy
import os
import re
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dir", type=str, required=True,
        help="Directory with channels")
    parser.add_argument(
        "-at_least", type=int, required=True,
        help="Minimum number of *.jp2 in each subfolder")
    parser.add_argument(
        "-seed", type=str, required=True,
        help="File with seed for choosing of file to replicate")
    args = parser.parse_args()

    numpy.random.seed(numpy.fromfile(args.seed, dtype=numpy.int32, count=1024))

    first = True
    jp2 = re.compile("\.jp2$", re.IGNORECASE)
    for basedir, dirlist, filelist in os.walk(args.dir, topdown=False):
        found_files = []
        for nme in filelist:
            if jp2.search(nme) is not None:
                found_files.append("%s/%s" % (basedir, nme))
        n = len(found_files)
        if n >= args.at_least or n == 0:
            continue
        print("Will replicate some of %d files in %s up to %d" % (
            n, basedir, args.at_least))
        if first:
            first = False
            k = 15
            for kk in range(k, 0, -1):
                print("Will do the rest after %d seconds" % (kk))
                time.sleep(1)
            print("Will replicate now")
        for i in range(args.at_least - n):
            ii = numpy.random.randint(n)
            nme = found_files[ii]
            shutil.copy(nme, "%s_%d.jp2" % (nme[:-4], i))

    print("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
