"""
Created on Nov 7, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import os


def update(root):
    root.common.engine.source_dirs.append(os.path.dirname(__file__))
