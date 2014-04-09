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

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "cifar"},
               "global_alpha": 0.1,
               "global_lambda": 0.00005,
               "layers": [100, 10],
               "loader": {"minibatch_maxsize": 180},
               "n_bars": 50,
               "path_for_out_data": os.path.join(root.common.cache_dir,
                                                 "tmp/"),
               "path_for_train_data": "/data/veles/cifar/10",
               "path_for_valid_data": "/data/veles/cifar/10/test_batch",
               "weights_plotter": {"limit": 25}
               }
