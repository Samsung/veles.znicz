"""
Created on Nov 7, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import os


def update(root):
    root.common.ocl_dirs.append(os.path.join(os.path.dirname(__file__), "ocl"))
