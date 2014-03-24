"""
Created on May 21, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


import os
from veles.config import root

root.common.ocl_dirs.append(os.path.join(os.path.dirname(__file__), "ocl"))
