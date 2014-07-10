"""
Created on Jul 8, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from collections import namedtuple
import numpy
from numpy.linalg import norm
import scipy.signal
import scipy.stats


from veles.znicz.nn_plotting_units import Weights2D


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
    peak_C = corr_S // 2
    maxdist = numpy.sqrt(2) * peak_C
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
    threshold = numpy.max([
        numpy.min([0.95, mean + stddev * params.magnitude_threshold]), 0.75])
    mask *= sub_matrix > threshold

    # Filter by peak sharpness
    vals = kurt_matrix[numpy.logical_not(numpy.isnan(kurt_matrix))]
    mean = numpy.mean(vals)
    stddev = numpy.std(vals)
    kurt_matrix[numpy.isnan(kurt_matrix)] = numpy.min(vals)
    mask *= kurt_matrix > mean + stddev * params.peak_threshold

    # Filter by x-correlation argmax distance from the center
    vals = corr_matrix[corr_matrix > 0]
    mean = numpy.mean(vals)
    stddev = numpy.std(vals)
    threshold = numpy.max([
        numpy.min([0.95, mean + stddev * params.form_threshold]), 0.8])
    mask *= corr_matrix > threshold

    # Fix boundary='symm' symmetry violation
    for x in range(N):
        for y in range(N):
            if x == y:
                continue
            if mask[x, y] > 0 and mask[y, x] == 0:
                mask[x, y] = 0
    del corr_matrix
    del sub_matrix
    del kurt_matrix

    # Find cliques in the similarity graph.
    # We use Bron-Kerbosch algorithm.
    # http://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm.
    similar_sets = []
    visited = set()
    for x in range(N):
        if x in visited:
            continue
        stack = [x]
        clique = {x}
        wrong = set()
        while len(stack):
            cx = stack.pop()
            for y in range(N):
                if mask[cx, y] == 0 or y in wrong:
                    continue
                good = True
                for z in clique:
                    if mask[y, z] == 0:
                        good = False
                        wrong.add(y)
                        break
                if good:
                    clique.add(y)
                    stack.append(y)
        if len(clique) > 1:
            similar_sets.append(clique)
            visited.update(clique)
    return similar_sets


class SimilarWeights2D(Weights2D):
    def __init__(self, workflow, **kwargs):
        kwargs['split_channels'] = False
        super(SimilarWeights2D, self).__init__(workflow, **kwargs)
        self.form_threshold = kwargs.get('form_threshold', 1.1)
        self.peak_threshold = kwargs.get('peak_threshold', 0.5)
        self.magnitude_threshold = kwargs.get('magnitude_threshold', 0.65)

    def prepare_pics(self, inp, transposed):
        if self.transposed:
            inp = inp.transpose()
        n_channels, _, _ = self.get_number_of_channels(inp)
        sims = get_similar_kernels(
            inp, n_channels,
            SimilarityCalculationParameters(
                self.form_threshold, self.peak_threshold,
                self.magnitude_threshold))
        self.info("Founf similar kernels: %s", str(sims))
        siminp = numpy.empty((sum([len(s) for s in sims]), inp.shape[1]),
                             dtype=inp.dtype)
        counter = 0
        for simset in sims:
            for s in simset:
                siminp[counter] = inp[s]
                counter += 1
        if counter == 0:
            return []
        return super(SimilarWeights2D, self).prepare_pics(siminp, False)
