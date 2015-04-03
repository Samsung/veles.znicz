"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jul 9, 2014

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
