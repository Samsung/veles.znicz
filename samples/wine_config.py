#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Wine config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


import os

from veles.config import root, Config
import veles.rnd as rnd


root.decision = Config()  # not necessary for execution (it will do it in real
# time any way) but good for Eclipse editor

# optional parameters

root.common.update = {"plotters_disabled": True}

root.update = {"decision": {"fail_iterations": 200,
                            "snapshot_prefix": "wine"},
               "loader": {"minibatch_maxsize": 1000000,
                          "rnd": rnd.default,
                          "view_group": "LOADER"},
               "wine": {"global_alpha": 0.5,
                        "global_lambda": 0.0,
                        "layers": [8, 3],
                        "path_for_load_data":
                        os.path.join(root.common.veles_dir,
                                     "veles/znicz/samples/wine/wine.data")}}
