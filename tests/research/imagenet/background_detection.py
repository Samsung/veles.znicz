# encoding: utf-8
'''
A test script for is_background function
'''

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
