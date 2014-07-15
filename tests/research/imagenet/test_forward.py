"""
Created on Jul 11, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import numpy
import unittest

from veles.tests import DummyWorkflow
from veles.znicz.tests.research.imagenet.forward import ForwardStage1Loader


class TestForward1(unittest.TestCase):
    def setUp(self):
        self.loader = ForwardStage1Loader(DummyWorkflow(), "", "")
        self.loader.aperture = 256

    def test_intersects(self):
        self.assertTrue(self.loader._intersects(
            (0, 0), numpy.array([[-100, -100], [100, -100],
                                 [100, 100], [-100, 100]])))
        self.assertFalse(self.loader._intersects(
            (100, 50), numpy.array([[500, -100], [600, 0],
                                    [500, 100], [400, 0]])))
        self.assertTrue(self.loader._intersects(
            (33, 66), numpy.array([[50, -150], [150, 0],
                                   [0, 100], [-100, -50]])))
        self.assertFalse(self.loader._intersects(
            (33, 66), numpy.array([[50, -200], [150, -50],
                                   [0, 50], [-100, -100]])))

    def test_inside(self):
        self.assertTrue(self.loader._inside(
            (0, 0), numpy.array([[-100, -100], [100, -100],
                                 [100, 100], [-100, 100]])))
        self.assertFalse(self.loader._inside(
            (150, 150), numpy.array([[-100, -100], [100, -100],
                                     [100, 100], [-100, 100]])))
        self.assertFalse(self.loader._inside(
            (60, 10), numpy.array([[0, 50], [100, 50],
                                   [100, 150], [0, 150]])))
        self.assertTrue(self.loader._inside(
            (60, 60), numpy.array([[0, 50], [100, 50],
                                   [100, 150], [0, 150]])))

    def test_intersection_area(self):
        self.loader.aperture = 100
        area = self.loader._calculate_approximate_area_of_intersection(
            (0, 0), numpy.array([[0, 50], [100, 50], [100, 150], [0, 150]]))
        self.assertGreater(area, 0.45)
        self.assertLess(area, 0.55)

    def test_calculate_number_of_variants(self):
        nvars = self.loader._calculate_number_of_variants(
            (640, 480), 2 * numpy.pi / 16, 10, 0.5)
        self.assertEqual(14130, nvars)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testIntersects']
    unittest.main()
