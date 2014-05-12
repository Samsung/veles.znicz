#!/usr/bin/python3.3 -O
"""
Created on Mart 26, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os
from veles.config import root, Config

root.all2all = Config()  # not necessary for execution (it will do it in real
root.decision = Config()  # time any way) but good for Eclipse editor
root.loader = Config()

# optional parameters

root.gtzan.labels = {"blues": 0,
                     "country": 1,
                     "jazz": 2,
                     "pop": 3,
                     "rock": 4,
                     "classical": 5,
                     "disco": 6,
                     "hiphop": 7,
                     "metal": 8,
                     "reggae": 9}
root.gtzan.features_shape = {"CRP": 12}

root.update = {"decision": {"fail_iterations": 100,
                            "snapshot_prefix": "gtzan"},
               "gtzan": {"exports":
                         ["features", "labels", "norm_add", "norm_mul"],
                         "features":
                         ["Energy", "Centroid", "Flux", "Rolloff",
                          "ZeroCrossings", "CRP"],
                         "learning_rate": 0.01,
                         "weights_decay": 0.00005,
                         "layers": [100, 500, 10],
                         "minibatch_maxsize": 108,
                         "minibatches_in_epoch": 1000,
                         "pickle_fnme":
                         os.path.join(root.common.test_dataset_root,
                                      "music/GTZAN/gtzan.pickle"),
                         "snapshot": "",
                         "window_size": 100}}
