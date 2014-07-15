"""
Created on Jul 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from collections import defaultdict, namedtuple
import cv2
import json
import logging
import math
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

    def __init__(self, workflow, images_json, labels_txt, **kwargs):
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
        self.labels_txt = labels_txt
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
        self.max_bboxes = 128
        self.minibatch_data = formats.Vector()
        self.minibatch_size = 0
        self.minibatch_bboxes = formats.Vector()
        self.minibatch_labels = formats.Vector()
        self._state = ()
        self._current_image = ""
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
        with open(self.labels_txt, "r") as txt:
            values = txt.read().split()
            self._labels_mapping = dict(values[1::2], map(int, values[::2]))
        self.aperture = 256  # FIXME: get it from self.entry
        channels = 4  # FIXME: get it from self.entry
        self.minibatch_data.mem = numpy.zeros(
            (self.max_batch_size, self.aperture ** 2 * channels),
            dtype=self.entry.weights.mem.dtype)
        self.minibatch_bboxes.mem = numpy.zeros(
            (self.max_batch_size, self.max_bboxes, 4), dtype=numpy.uint16)
        self.minibatch_labels.mem = numpy.zeros(
            (self.max_batch_size, self.max_bboxes), dtype=numpy.int32)

        self._image_iter = iter(self.images.keys())
        self._set_next_state()

        if device is None:
            return

        self.minibatch_data.initialize(device)
        self.minibatch_bboxes.initialize(device)
        self.minibatch_labels.initialize(device)

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
                bbox, _ = self._transform_shape(shape, angle, scale)
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

    def _transform_shape(self, shape, angle, scale):
        bbox = numpy.array([[0, 0], [shape[1], 0],
                            [shape[1], shape[0]], [0, shape[0]]])
        bbox = self._transform_bbox(bbox, angle, scale)
        dxdy = [numpy.min(bbox[:, dim]) for dim in (0, 1)]
        for dim in (0, 1):
            bbox[:, dim] -= dxdy[dim]
        return bbox, dxdy

    def _transform_bbox(self, bbox, angle, scale):
        matrix = scale * numpy.array(
            [[numpy.cos(angle), -numpy.sin(angle)],
             [numpy.sin(angle), numpy.cos(angle)]])
        return bbox.dot(matrix)

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
        try:
            image_data, angle, scale, bbox, x, y = self._state
            x, y = self._set_next_x_y(image_data, bbox, x, y)
            if x * y > 0:
                self._state = (image_data, angle, scale, bbox, x, y)
                return
            scale += self._real_scale_step
            if scale > self._max_scale:
                scale = self._min_scale
                angle += self._real_angle_step
        except ValueError:
            angle = 2 * numpy.pi
        if angle >= 2 * numpy.pi:
            angle = 0
            self._load_next_image()
            scale = self._min_scale
        bbox, image_data = self._transform_image(angle, scale)
        self._set_next_x_y(image_data, bbox, x, y)
        self._state = (image_data, angle, scale, bbox, x, y)

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
        self._current_image = next(self._image_iter)
        file_name = self.images[self._current_image]["path"]
        self._original_image_data = self.decode_image(file_name)
        self._set_number_of_variants()
        self._min_scale, self._max_scale = self._calculate_scale_min_max(
            self._original_image_data.shape)

    def _set_number_of_variants(self):
        angle_step = self.angle_step
        scale_steps = self.scale_steps
        overlap_factor = self.overlap_factor
        nvars = self.max_batch_size
        while nvars > self.max_batch_size:
            nvars = self._calculate_number_of_variants(
                self._original_image_data.shape, angle_step, scale_steps,
                overlap_factor)
            if nvars <= self.max_batch_size:
                break
            if angle_step > numpy.pi / 6:
                angle_step = numpy.pi / 6
                continue
            if scale_steps > 10:
                scale_steps = 10
                continue
            if overlap_factor < 0.25:
                overlap_factor = 0.25
                continue
            self.warning("Applied MEDIUM quality for %s", self._current_image)
            if angle_step > numpy.pi / 4:
                angle_step = numpy.pi / 4
                continue
            if scale_steps > 5:
                scale_steps = 5
                continue
            if overlap_factor < 0.5:
                overlap_factor = 0.5
                continue
            self.warning("Applied LOW quality for %s", self._current_image)
            if angle_step > numpy.pi / 2:
                angle_step = numpy.pi / 2
                continue
            if scale_steps > 2:
                scale_steps = 2
                continue
            self.error("Debug this image: %s", self._current_image)
            raise NotImplementedError()
        self.info("Will process %d transformations of %s",
                  nvars, self._current_image)
        return nvars

    def _transform_image(self, angle, scale, add_sobel_channel=True):
        """Transform original image by rotating and scaling, and also adding
        alpha- and Sobel-channels (optionally).

        Args:
            image: 3D numpy array (shape=(height, width, colors_num);
                dtype=numpy.uint8)
            angle: rotation angle in radians
            scale: scaling parameter (1.0 means no scaling)
            add_sobel_channel: flag specifying adding Sobel-channel
                to output image

        Returns:
            Tuple of out_image and out_bbox.

            out_image: 3D numpy array in RGBA[S] format with image after
                rotation and scaling (RGBA + Sobel-channel (optionally))
                (shape=(img_side, img_side, colors_num + 2),
                where img_side = 2 * max(height, width); dtype=numpy.uint8).
            out_bbox: bounding box after rotation and scaling (list of
                vertexes: [(x0, y0), (x1, y1), (x2, y2), (x3, y3)])
        """
        assert len(self._original_image_data.shape) == 3, "Bad image shape"

        height = self._original_image_data.shape[0]
        width = self._original_image_data.shape[1]
        colors_num = self._original_image_data.shape[-1]

        # get grayscale image
        if colors_num > 1:
            gray_img = cv2.cvtColor(self._original_image_data,
                                    cv2.COLOR_RGB2GRAY)
        else:
            gray_img = self._original_image_data

        # calculate optimal output image size
        rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2),
                                             angle * 180 / math.pi, scale)
        bbox = [tuple(rot_matrix.dot(numpy.array([0, 0, 1]))),
                tuple(rot_matrix.dot(numpy.array([width, 0, 1]))),
                tuple(rot_matrix.dot(numpy.array([width, height, 1]))),
                tuple(rot_matrix.dot(numpy.array([0, height, 1])))]
        out_width = (max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]) -
                     min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]))
        out_width = max(out_width, width)
        out_height = (max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]) -
                      min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]))
        out_height = max(out_height, height)
        logging.debug("optimal output image size: %dx%d", out_width,
                      out_height)

        # calculate bounding box after rotation and scaling
        center = (out_width / 2, out_height / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle * 180 / math.pi,
                                             scale)
        x0 = (out_width - width) / 2
        y0 = (out_height - height) / 2
        out_bbox = [tuple(rot_matrix.dot(numpy.array([x0, y0, 1]))),
                    tuple(rot_matrix.dot(numpy.array([x0 + width, y0, 1]))),
                    tuple(rot_matrix.dot(numpy.array([x0 + width,
                                                      y0 + height, 1]))),
                    tuple(rot_matrix.dot(numpy.array([x0, y0 + height, 1])))]
        logging.debug("output bounding box: %s", out_bbox)

        # initialize image with Sobel-channel
        s_shape = (out_height, out_width)
        s_img = numpy.zeros(shape=s_shape, dtype=numpy.uint8)

        if add_sobel_channel:
            # get results of Sobel-filtration in x- and y-directions
            sobel_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=5)

            # join sobel_x and sobel_y and clip to uint8
            sobel = numpy.sqrt(numpy.square(sobel_x) + numpy.square(sobel_y))
            sobel -= sobel.min()
            max_val = sobel.max()
            if max_val:
                sobel *= 255.0 / max_val
            sobel = numpy.clip(sobel, 0, 255).astype(numpy.uint8)

            # fill Sobel-channel
            for (i, j), _ in numpy.ndenumerate(sobel):
                out_i = center[1] - height / 2 + i
                out_j = center[0] - width / 2 + j
                s_img[out_i, out_j] = sobel[i, j]

            # rotation and scaling for Sobel-channel
            s_img = cv2.warpAffine(s_img, rot_matrix, (s_img.shape[1],
                                                       s_img.shape[0]))

        # fill RGB color values to expanded image and add alpha-channel
        rgba_shape = (out_height, out_width, colors_num + 1)
        rgba_img = numpy.zeros(shape=rgba_shape, dtype=numpy.uint8)
        for i, j in ((i, j) for i in range(height) for j in range(width)):
            out_i = center[1] - height / 2 + i
            out_j = center[0] - width / 2 + j
            for c in range(colors_num):
                rgba_img[out_i, out_j, c] = self._original_image_data[i, j, c]
            # set alpha-channel to 255 as default value
            rgba_img[out_i, out_j, colors_num] = 255

        # rotation and scaling for rgba_img
        rgba_img = cv2.warpAffine(rgba_img, rot_matrix, (rgba_img.shape[1],
                                                         rgba_img.shape[0]))

        # add S-channels to RGBA image (if necessary)
        if add_sobel_channel:
            out_shape = (rgba_img.shape[0], rgba_img.shape[1], colors_num + 2)
            out_img = numpy.empty(shape=out_shape, dtype=numpy.uint8)
            for i, j in ((i, j) for i in range(rgba_img.shape[0])
                         for j in range(rgba_img.shape[1])):
                for c in range(colors_num + 1):
                    out_img[i, j, c] = rgba_img[i, j, c]
                out_img[i, j, colors_num + 1] = s_img[i, j]
            logging.debug("shape of output image: %s", out_img.shape)
            return out_img, out_bbox
        else:
            logging.debug("shape of output image: %s", rgba_img.shape)
            return rgba_img, out_bbox

    def _get_bbox_from_json(self, bbox):
        angle, x, y, width, height = [
            float(bbox[l]) for l in ('angle', 'x', 'y', 'width', 'height')]
        hwidth = width / 2
        hheight = height / 2
        bbox = numpy.array(((x - hwidth, y - hheight),
                            (x + hwidth, y - hheight),
                            (x + hwidth, y + hheight),
                            (x - hwidth, y + hheight)))
        return self._transform_bbox(bbox, angle, 1.0)

    def _get_label_from_json(self, label):
        return self._labels_mapping[label]

    def _create_bbox(self, transformed_bbox):
        return numpy.array(
            [numpy.min(transformed_bbox[:, i]) for i in (0, 1)] +
            [numpy.max(transformed_bbox[:, i]) for i in (0, 1)],
            dtype=numpy.uint16)

    def cpu_run(self):
        self.ocl_run()

    def ocl_run(self):
        self.minibatch_data.map_invalidate()
        self.minibatch_bboxes.map_invalidate()

        self.minibatch_size = self.max_minibatch_size
        for index in range(self.max_minibatch_size):
            transformed_image_data, angle, scale, _, _, _ = self._state
            self.minibatch_data.mem[index] = transformed_image_data
            if self.substage == 1:
                _, dxdy = self._transform_shape(
                    self._original_image_data.shape, angle, scale)
                meta = self.images[self._current_image]
                for bbindex, bbox in enumerate(meta["bbxs"]):
                    bbox = self._get_bbox_from_json(bbox)
                    bbox = self._transform_bbox(bbox, angle, scale)
                    for i in (0, 1):
                        bbox[:, i] -= dxdy[i]
                    self.minibatch_bboxes.mem[index, bbindex] = \
                        self._create_bbox(bbox)
                    self.minibatch_labels.mem[index, bbindex] = \
                        self._get_label_from_json(bbox["label"])
            else:
                self.minibatch_labels.mem[index, 0] = \
                    self._get_label_from_json(
                        self.images[self._current_image]["label"])
            try:
                self._set_next_state()
            except StopIteration:
                self.minibatch_size = index + 1
                break
