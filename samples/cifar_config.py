#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Mnist config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import os

from veles.config import root, Config

root.all2all = Config()  # not necessary for execution (it will do it in real
root.decision = Config()  # time any way) but good for Eclipse editor
root.loader = Config()

# optional parameters

train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
validation_dir = os.path.join(root.common.test_dataset_root,
                              "cifar/10/test_batch")

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "cifar"},
               "image_saver": {"out_dirs":
                               [os.path.join(root.common.cache_dir,
                                             "tmp/test"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/validation"),
                                os.path.join(root.common.cache_dir,
                                             "tmp/train")]},
               "loader": {"minibatch_maxsize": 180},
               "accumulator": {"n_bars": 30},
               "weights_plotter": {"limit": 25},
               "cifar": {"global_alpha": 0.1,
                         "global_lambda": 0.00005,
                         "layers": [100, 10],
                         "path_for_load_data": {"train": train_dir,
                                                "validation":
                                                validation_dir}}}
