#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Kanji config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
import six

from veles.config import root, Config

root.decision = Config()  # not necessary for execution (it will do it in real
root.loader = Config()  # time any way) but good for Eclipse editor

# optional parameters

root.update = {"decision": {"fail_iterations": 1000,
                            "snapshot_prefix": "kanji",
                            "store_samples_mse": True},
               "loader": {"minibatch_maxsize": 5103,
                          "validation_procent": 0.15},
               "weights_plotter": {"limit": 16},
               "kanji": {"global_alpha": 0.001,
                         "global_lambda": 0.00005,
                         "layers": [5103, 2889, 24 * 24],
                         "path_for_load_data":
                         {"target":
                          os.path.join(root.common.test_dataset_root,
                                       ("kanji/target/targets.%d.pickle" %
                                        (3 if six.PY3 else 2))),
                          "train":
                          os.path.join(root.common.test_dataset_root,
                                       "kanji/train")}}}
root.kanji.index_map = os.path.join(root.kanji.path_for_load_data.train,
                                    "index_map.%d.pickle" %
                                    (3 if six.PY3 else 2))
