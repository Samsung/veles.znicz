"""
Created on Jul 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from collections import defaultdict, namedtuple
import json
import numpy
from zope.interface import implementer

from veles import OpenCLUnit
import veles.formats as formats
from veles.znicz.tests.research.imagenet.processor import Processor
from veles.opencl_units import IOpenCLUnit


ForwardStage1LoaderState = namedtuple('ForwardStage1Loader',
                                      ['image', 'angle', 'scale', 'position'])


@implementer(IOpenCLUnit)
class ForwardStage1Loader(OpenCLUnit, Processor):
    """
    Imagenet loader for first processing stage.
    """

    def __init__(self, workflow, images_json, **kwargs):
        """
        kwargs:
            angle_step        the step with which rotate images
                              (not guaranteed)
            scale_steps       the number of scales images are tested
                              (not guaranteed)
            min_real_size     the minimal size of the image part to magnify
            overlap_factor    the amount of overlapping, the relative distance
                              between successive samples
            max_batch_size    the maximal overall amount of image
                              transformations
            minibatch_size    the maximal size of one minibatch
        """
        super(ForwardStage1Loader, self).__init__(workflow, **kwargs)
        self.images_json = images_json
        self.angle_step = kwargs.get('angle_step', 2 * numpy.pi / 36)
        self.scale_steps = kwargs.get('scale_steps', 10)
        self.max_batch_size = kwargs.get('max_batch_size', 4000)
        self.max_minibatch_size = kwargs.get('minibatch_size', 40)
        self.min_real_size = kwargs.get('min_real_size', (64, 0.05))
        self.overlap_factor = kwargs.get('overlap_factor', 0.5)
        self.min_intersection_area = kwargs.get('min_intersection_area', 0.5)
        assert 0 < self.overlap_factor <= 1
        self.images = {}
        self.bbox_mapping = defaultdict(list)
        self.no_bbox_mapping = defaultdict(list)
        self.aperture = 0
        self.substage = 1  # 1 or 2
        self.minibatch_data = formats.Vector()
        self.minibatch_size = 0
        self.batch_bboxes = formats.Vector()
        self._state = ()
        self._original_image_data = None
        self._image_iter = None
        self._min_scale = 0
        self._max_scale = 0
        self._real_scale_step = 0
        self._real_angle_step = 0
        self._real_overlap_factor = 0
        self.demand("entry")  # first forward unit

    def init_unpickled(self):
        super(ForwardStage1Loader, self).init_unpickled()

    def initialize(self, device, **kwargs):
        super(ForwardStage1Loader, self).initialize(**kwargs)
        for set_type in ("test", "validation", "train"):
            images_json = self.images_json % set_type
            try:
                self.info("Loading images JSON from %s" % images_json)
                with open(images_json, 'r') as fp:
                    self.images.update(json.load(fp))
            except:
                self.exception("Failed to load %s", images_json)
            for image_name, meta in sorted(self.images[set_type].items()):
                bboxes = meta["bbxs"]
                for bbox in bboxes:
                    label = bbox["label"]
                    self.bbox_mapping[label].append(image_name)
                if len(bboxes) == 0:
                    self.no_bbox_mapping[label].append(image_name)
        self.aperture = 256  # FIXME: get it from self.entry
        channels = 4  # FIXME: get it from self.entry
        self.minibatch_data.mem = numpy.zeros(
            (self.max_batch_size, self.aperture ** 2 * channels),
            dtype=self.entry.weights.mem.dtype)
        self.minibatch_bboxes.mem = numpy.zeros(4 * self.max_batch_size,
                                                dtype=numpy.int32)

        if device is None:
            return

        self.batch_data.initialize(device)
        self.batch_bboxes.initialize(device)

    def _calculate_scale_min_max(self, shape):
        maxsize = numpy.max(shape[:2])
        min_scale = self.aperture / maxsize
        min_real_size = numpy.max((
            self.min_real_size[0], maxsize * self.min_real_size[1]))
        max_scale = numpy.max((self.aperture / min_real_size, min_scale))
        return min_scale, max_scale

    def _calculate_number_of_variants(self, shape, angle_step, scale_steps,
                                      overlap_step):
        res = 0
        eps = numpy.finfo(float).eps
        min_scale, max_scale = self._calculate_scale_min_max(shape)
        scale_step = (max_scale - min_scale) / scale_steps
        for scale in numpy.arange(min_scale, max_scale + eps, scale_step):
            for angle in numpy.arange(0, 2 * numpy.pi, angle_step):
                rotscmat = scale * numpy.array(
                    [[numpy.cos(angle), -numpy.sin(angle)],
                     [numpy.sin(angle), numpy.cos(angle)]])
                bbox = numpy.array([[0, 0], [shape[1], 0],
                                    [shape[1], shape[0]], [0, shape[0]]])
                bbox = bbox.dot(rotscmat)
                for dim in (0, 1):
                    bbox[:, dim] -= numpy.min(bbox[:, dim])
                bbox_width, bbox_height = [
                    numpy.max((numpy.max(bbox[:, i]), self.aperture))
                    for i in (0, 1)]
                for x in numpy.arange(0, bbox_width - self.aperture + eps,
                                      self.aperture * self.overlap_factor):
                    for y in numpy.arange(0, bbox_height - self.aperture + eps,
                                          self.aperture * self.overlap_factor):
                        if self._check_aperture_payload(x, y, bbox):
                            res += 1
        return res

    def _check_aperture_payload(self, x, y, bbox):
        return self._intersects((x, y), bbox) and \
            self._calculate_approximate_area_of_intersection(
                (x, y), bbox) >= self.min_intersection_area

    def _intersects(self, lb, bbox):
        # Based on Separating Axis Theorem
        A = numpy.array([[lb[0], lb[1]], [lb[0] + self.aperture, lb[1]],
                         [lb[0] + self.aperture, lb[1] + self.aperture],
                         [lb[0], lb[1] + self.aperture]])
        for axis in (A[2] - A[3], A[2] - A[1],
                     bbox[3] - bbox[0], bbox[3] - bbox[2]):
            projA = numpy.dot(numpy.dot(A, axis).reshape(4, 1) * axis, axis)
            projB = numpy.dot(numpy.dot(bbox, axis).reshape(4, 1) * axis, axis)
            minA, minB = [numpy.min(p) for p in (projA, projB)]
            maxA, maxB = [numpy.max(p) for p in (projA, projB)]
            if minB > maxA or maxB < minA:
                return False
        return True

    def _inside(self, p, bbox):
        AB = bbox[1] - bbox[0]
        AP = numpy.array(p) - bbox[0]
        BC = bbox[2] - bbox[1]
        BP = numpy.array(p) - bbox[1]

        return 0 <= numpy.dot(AB, AP) <= numpy.dot(AB, AB) and \
            0 <= numpy.dot(BC, BP) <= numpy.dot(BC, BC)

    def _calculate_approximate_area_of_intersection(self, lp, bbox):
        if self._inside(lp, bbox) and \
           self._inside((lp[0] + self.aperture, lp[1]), bbox) and \
           self._inside((lp[0] + self.aperture,
                        lp[1] + self.aperture), bbox) and \
           self._inside((lp[0], lp[1] + self.aperture), bbox):
            return 1.0
        overall = 0
        inside = 0
        step = self.aperture / 10
        for x in numpy.arange(lp[0], lp[0] + self.aperture + step, step):
            for y in numpy.arange(lp[1], lp[1] + self.aperture, step):
                inside += self._inside((x, y), bbox)
                overall += 1
        return inside / overall

    def _set_next_state(self):
        transformed_image_data, angle, scale, bbox, x, y = self._state
        x, y = self._set_next_x_y(transformed_image_data, bbox, x, y)
        if x * y > 0:
            self._state = (transformed_image_data, angle, scale, bbox, x, y)
            return
        scale += self._real_scale_step
        if scale > self._max_scale:
            scale = self._min_scale
            angle += self._real_angle_step
        if angle >= 2 * numpy.pi:
            angle = 0
            self._load_next_image()
            scale = self._min_scale
        bbox = self._transform_image(angle, scale)
        self._set_next_x_y(transformed_image_data, bbox, x, y)
        self._state = (transformed_image_data, angle, scale, bbox, x, y)

    def _set_next_x_y(self, transformed_image_data, bbox, x, y):
        while x + self.aperture <= transformed_image_data.shape[1] and \
                not self.check_aperture_payload(x, y, bbox):
            x += self.aperture * self._real_overlap_factor
        if x + self.aperture <= transformed_image_data.shape[1]:
            return x, y
        x = 0
        while y + self.aperture <= transformed_image_data.shape[0] and \
                not self.check_aperture_payload(x, y, bbox):
            y += self.aperture * self._real_overlap_factor
        if y + self.aperture <= transformed_image_data.shape[0]:
            return x, y
        y = 0
        return x, y

    def _load_next_image(self):
        next_image = self._image_iter.__next__()
        file_name = self.images[next_image]["path"]
        self._original_image_data = self.decode_image(file_name)
        self._set_number_of_variants()

    def _set_number_of_variants(self):
        angle_step = self.angle_step
        scale_steps = self.scale_steps
        overlap_factor = self.overlap_factor
        nvars = self.max_batch_size
        while nvars > self.max_batch_size:
            nvars = self._calculate_number_of_variants(
                self._original_image_data.shape, angle_step, scale_steps,
                overlap_factor)
            if angle_step > numpy.pi / 6:
                angle_step = numpy.pi / 6
                continue
            if angle_step > numpy.pi / 4:
                angle_step = numpy.pi / 4
                continue
            if scale_steps > 10:
                scale_steps = 10
                continue
            if scale_steps > 5:
                scale_steps = 5
                continue
            if overlap_factor < 0.25:
                overlap_factor = 0.25
                continue
            if overlap_factor < 0.5:
                overlap_factor = 0.5
                continue
            raise NotImplementedError("No idea what to do in this situation")

    def _transform_image(self, angle, scale):
        pass

    def cpu_run(self):
        pass

    def ocl_run(self):
        transformed_image_data, angle, scale, bbox, x, y = self._state
