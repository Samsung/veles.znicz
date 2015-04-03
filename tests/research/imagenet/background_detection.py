# encoding: utf-8
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on July 30, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""

try:
    import cv2
except ImportError:
    import warnings
    warnings.warn("Failed to import OpenCV bindings")
import numpy
import scipy.stats
import statsmodels as sm


def is_background(pic_path, thr=8.0):
    """
    Reads an image in grayscale, then fits its color intensity distribution
        as normal. Then compares fitted CDF with empirical CDF. If they are
        similar, thinks, that it is background

    Args:
        pic_path(str): path to image
        the(float): a threshold
    Returns:
        bool

    """
    pic_ravel = cv2.imread(pic_path, 0).ravel()
    mu, std = scipy.stats.norm.fit(pic_ravel)
    x_array = numpy.linspace(0, 255, num=256)
    cdf = scipy.stats.norm.cdf(x_array, mu, std)
    ecdf = sm.tools.tools.ECDF(pic_ravel)(x_array)

    delta = numpy.sum(numpy.abs(ecdf - cdf))

    return delta < thr
