"""
Created on Jul 8, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from collections import namedtuple
import numpy
from numpy.linalg import norm
import scipy.signal
import scipy.stats


SimilarityCalculationParameters = namedtuple(
    'SimilarityCalculationParameters', ['form_threshold', 'peak_threshold',
                                        'magnitude_threshold'])


def get_similar_kernels(weights, channels=3,
                        params=SimilarityCalculationParameters(1.1, .5, .65)):
    # number of neurons
    N = weights.shape[0]
    # number of weights in each channel
    S = int(numpy.sqrt(weights.shape[1] / channels))
    # size of the x-correlation matrix between two SxS matrices
    corr_S = S * 2 - 1
    peak_delta = corr_S * params.peak_radius
    peak_C = corr_S // 2
    maxdist = numpy.sqrt(2) * peak_C
    # minimal and maximal peak position
    min_peak = max_peak = peak_C
    min_peak -= peak_delta
    max_peak += peak_delta
    parts = [weights[:, c::channels] for c in range(channels)]
    # the following matrices will be filled in for-s
    corr_matrix = numpy.empty((N, N))
    sub_matrix = numpy.empty((N, N))
    kurt_matrix = numpy.empty((N, N))
    # compare each with each
    for x in range(N):
        for y in range(N):
            if x == y:
                corr_matrix[x, y] = sub_matrix[x, y] = 0
                kurt_matrix[x, y] = numpy.NAN
                continue

            corr = numpy.zeros((corr_S, corr_S))
            for ch in parts:
                corr += scipy.signal.correlate2d(ch[x].reshape(S, S),
                                                 ch[y].reshape(S, S),
                                                 boundary='symm')
            amx, amy = numpy.unravel_index(numpy.argmax(corr), corr.shape)
            dist = numpy.sqrt((amx - peak_C) ** 2 + (amy - peak_C) ** 2)
            corr_matrix[x, y] = 1 - dist / maxdist
            kurt_matrix[x, y] = scipy.stats.kurtosis(corr.ravel(), bias=False)

            diff = 0
            for ch in parts:
                delta = ch[x] - ch[y]
                delta = norm(delta)
                diff += delta * delta
            diff = numpy.sqrt(diff)
            sub_matrix[x, y] = 1 - diff

    # the indices of similar kernels
    mask = numpy.ones((N, N))
    # Filter by normalized difference
    vals = sub_matrix[sub_matrix > 0]
    mean = numpy.mean(vals)
    stddev = numpy.std(vals)
    mask *= sub_matrix > mean + stddev * params.magnitude_threshold

    # Filter by peak sharpness
    vals = kurt_matrix[not numpy.isnan(kurt_matrix)]
    mean = numpy.mean(vals)
    stddev = numpy.std(vals)
    mask *= kurt_matrix > mean + stddev * params.peak_threshold

    # Filter by x-correlation argmax distance from the center
    vals = corr_matrix[corr_matrix > 0]
    mean = numpy.mean(vals)
    stddev = numpy.std(vals)
    mask *= corr_matrix > mean + stddev * params.form_threshold

    # Fix boundary='symm' symmetry violation
    for x in range(N):
        for y in range(N):
            if x == y:
                continue
            if mask[x, y] > 0 and mask[y, x] == 0:
                mask[x, y] = 0

    # Find components of strong connectivity
    # TODO(v.markovtsev): the algorithm is described at
    # http://e-maxx.ru/algo/strong_connected_components
    return None
