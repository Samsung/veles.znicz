"""
Created on Jul 11, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import numpy
import unittest

from veles.tests import DummyWorkflow
from veles.znicz.tests.research.imagenet.forward_loader import \
    ImagenetForwardLoaderBbox


class TestForward1(unittest.TestCase):
    def setUp(self):
        self.loader = ImagenetForwardLoaderBbox(DummyWorkflow(), "", "", "")
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

    def test_transform_image(self):
        img = numpy.zeros((100, 200, 3), dtype=numpy.uint8)
        img[:5, :, 0] = 255
        img[-5:, :, 0] = 255
        img[:, :5, 0] = 255
        img[:, -5:, 0] = 255
        self.loader._original_image_data = img
        transformed, bbox = self.loader._transform_image(numpy.pi / 6, 2.0)
        self.assertEqual(len(transformed.shape), 3)
        self.assertEqual(transformed.shape[2], 5)
        self.assertEqual(len(bbox), 4)
        transformed, _ = self.loader._transform_image(numpy.pi / 6, 0.5)
        self.assertEqual(len(transformed.shape), 3)
        self.assertEqual(transformed.shape[2], 5)
        """
        from PIL import Image
        img = Image.fromarray(transformed[:, :, :4], 'RGBA')
        img.save("transformed.png")
        """

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testIntersects']
    # unittest.main()

    import os
    import veles.formats as formats
    base = "/data/veles/datasets/imagenet/2014_img_split_0/1"
    loader = ImagenetForwardLoaderBbox(
        DummyWorkflow(),
        os.path.join(base, "images_imagenet_1_img_%s_0.json"),
        os.path.join(base, "labels_int_1_img_0.txt"),
        os.path.join(base, "matrixes_1_img_0.pickle"))

    class LayerStub(object):
        def __init__(self):
            self.weights = formats.Vector()
            self.weights.mem = numpy.zeros((256, 256))

    loader.entry = LayerStub()
    loader.initialize(device=None)
    while True:
        loader.run()
        if loader.minibatch_size == 0:
            break
