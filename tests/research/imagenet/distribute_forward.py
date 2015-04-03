#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on July 30, 2014

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


import datetime
import pickle
import sys

if __name__ == "__main__":
    stats = []
    total = 0
    img_count = 0
    print("Loading %s..." % sys.argv[1])
    with open(sys.argv[1], "rb") as fin:
        while True:
            try:
                img = pickle.load(fin)[1]
                size = len(img["bbxs"])
                stats.append(size)
                total += size
                img_count += 1
            except EOFError:
                break
    print("Found %d images, %d bboxes" % (img_count, total))
    slaves = int(sys.argv[2])
    norm = total // slaves
    print("%d  bboxes for each slave" % norm)
    print("### ESTIMATED TIME: %s ###" % datetime.timedelta(
        seconds=(norm / 50000 * 1.5 * 3600)))
    print("")
    current_sum = 0
    border = 0
    minmaxs = []
    for index, count in enumerate(stats):
        current_sum += count
        if current_sum >= norm:
            print("[%d, %d)" % (border, index + 1))
            minmaxs.append((border, index + 1))
            border = index + 1
            current_sum = 0
    if border < len(stats):
        print("[%d, 0)" % border)
        minmaxs.append((border, 0))

    print('')
    print('-' * 80)
    print('')
    for index, minmax in enumerate(minmaxs):
        print(index)
        print('scripts/velescli.py -p "" -s -d 0:1 --debug MergeBboxes  '
              'veles/znicz/tests/research/imagenet/imagenet_forward.py - '
              'root.loader.min_index=%d root.loader.max_index=%d' % minmax)
