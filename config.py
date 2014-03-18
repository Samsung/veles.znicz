"""
Created on Jan 9, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""

import os

import veles.config as config


config.ocl_dirs.append(os.path.join(os.path.dirname(__file__), "ocl"))
