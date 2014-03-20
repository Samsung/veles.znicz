"""
Created on Jan 9, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""

import os

from veles.config import root


root.common.ocl_dirs.append(os.path.join(os.path.dirname(__file__), "ocl"))
