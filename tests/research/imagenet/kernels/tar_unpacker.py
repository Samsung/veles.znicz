"""
Created on Aug 13, 2014

Data archiving.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""
import tarfile
import os


def extract(tar_full_name, extract_path):
    """
    Extract  data from archive to extract_path/data folder
    Args:
        tar_full_name(str): archive name
        extract_path(str): path for extraction files
    """
    data_path = os.path.join(extract_path, 'data')
    if os.path.exists(data_path):
        return
    else:
        data_tar = tarfile.open(tar_full_name, 'r')
        data_tar.extractall(extract_path)
