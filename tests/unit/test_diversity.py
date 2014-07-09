"""
Created on Jul 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import numpy
import os
import unittest

from veles.znicz.diversity import get_similar_kernels


class Test(unittest.TestCase):
    def testSimilarSets(self):
        weights = numpy.load(os.path.join(os.path.dirname(__file__),
                                          'data/diversity_weights.npy'))
        sims = get_similar_kernels(weights)
        self.assertEqual(sims, [{1, 27, 4}, {18, 13}])
        """
        # Visualize a 2-D matrix
        from pylab import pcolor, show, colorbar, xticks, yticks
        pcolor(matrix)
        colorbar()
        xticks(numpy.arange(0.5, corr_matrix.shape[0] + 0.5),
                            range(0, corr_matrix.shape[0]))
        yticks(numpy.arange(0.5, corr_matrix.shape[1] + 0.5),
                            range(0, corr_matrix.shape[1]))
        show()
        """


if __name__ == "__main__":
    unittest.main()
