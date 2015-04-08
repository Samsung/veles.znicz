# -*-coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 12, 2014

Example of Kohonen map demo on a sample dataset.

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


import os
from veles.config import root

spam_dir = os.path.dirname(__file__)

root.spam_kohonen.update({
    "forward": {"shape": (8, 8)},
    "decision": {"epochs": 200},
    "loader": {"minibatch_size": 80,
               "force_numpy": True,
               "ids": True,
               "classes": False,
               "file": "/data/veles/VD/VDLogs/histogramConverter/data/hist",
               },
    "train": {"gradient_decay": lambda t: 0.002 / (1.0 + t * 0.00002),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.00002)},
    "exporter": {"file": "classified_fast4.txt"}})
