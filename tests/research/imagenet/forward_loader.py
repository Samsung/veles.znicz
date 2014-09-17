"""
Created on Jul 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import cv2
from collections import defaultdict
import datetime
import json
import math
import numpy
import os
import time
from twisted.internet import reactor
from zope.interface import implementer

from veles import OpenCLUnit
import veles.error as error
import veles.formats as formats
from veles.mutable import Bool
from veles.pickle2 import pickle
from veles.znicz.tests.research.imagenet.processor import Processor
from veles.opencl_units import IOpenCLUnit
from veles.external.progressbar.progressbar import ProgressBar, Percentage, Bar
from veles.workflow import NoMoreJobs


@implementer(IOpenCLUnit)
class ImagenetForwardLoaderBbox(OpenCLUnit, Processor):
    """
    Imagenet loader for the first processing stage.

    Defines:
        ended                Bool which signals when dataset is voer
        minibatch_data       actual data to apply forward propagation
        minibatch_bboxes     corresponding bboxes
        minibatch_images     list of tuples (path, shape) for each bbox
        minibatch_size       minibatch size
        current_image        current image file name (dict key)
    """

    HARDCODED_BBOXES = [(0.479, 0.598, 0.319, 0.213),
                        (0.454, 0.556, 0.501, 0.457),
                        (0.499, 0.606, 0.394, 0.3854),
                        (0.489, 0.518, 0.672, 0.717),
                        (0.465, 0.502, 0.294, 0.708),
                        (0.492, 0.489, 0.711, 0.447),
                        (0.503, 0.631, 0.4, 0.302)]

    def __init__(self, workflow, bboxes_file_name, **kwargs):
        kwargs["view_group"] = "LOADER"
        super(ImagenetForwardLoaderBbox, self).__init__(workflow, **kwargs)
        self.bboxes_file_name = bboxes_file_name
        self.aperture = 0
        self.channels = 0
        self.minibatch_data = formats.Vector()
        self.minibatch_size = 0
        self.minibatch_images = []
        self.max_minibatch_size = 0
        self.minibatch_bboxes = 0
        self.add_sobel = False
        self.angle_step = kwargs.get("angle_step", numpy.pi / 4)
        assert self.angle_step > 0
        self.max_angle = kwargs.get("max_angle", numpy.pi)
        self.min_angle = kwargs.get("min_angle", -numpy.pi)
        self._calc_angles()
        self.ended = Bool()
        self._state = (self.min_angle, False)  # angle, flip
        self.current_image = ""  # image file name == pickled dict's key
        self._current_image_data = None
        self._current_bbox = None  # used when another image appears
        self.mean = None
        self.total = 0
        self._progress = None
        self._initial_state = True
        self.mode = ""
        self.bboxes = {}
        self.only_this_file = kwargs.get("only_this_file", "")
        self.raw_bboxes_min_area = kwargs.get("raw_bboxes_min_area", 0)
        self.raw_bboxes_min_size = kwargs.get("raw_bboxes_min_size", 0)
        self.raw_bboxes_min_area_ratio = kwargs.get(
            "raw_bboxes_min_area_ratio", 0)
        self.raw_bboxes_min_size_ratio = kwargs.get(
            "raw_bboxes_min_size_ratio", 0)
        self.min_index = kwargs.get("min_index", 0)
        self.max_index = kwargs.get("max_index", 0)
        # entry is the first forward unit
        self.demand("entry_shape", "mean")

    def init_unpickled(self):
        super(ImagenetForwardLoaderBbox, self).init_unpickled()
        self._failed_minibatches = []
        self._pending_minibatches = defaultdict(list)
        self._progress_prevval = 0

    @property
    def current_image_size(self):
        return self._current_image_data.shape[:2] \
            if self._current_image_data is not None else 0

    def _calc_angles(self):
        self.angles = int(numpy.ceil((self.max_angle - self.min_angle +
                                      0.0001) / self.angle_step))
        self.info("Will rotate each bbox %d times", self.angles)

    def _load_bboxes(self):
        self.info("Loading bboxes from %s...", self.bboxes_file_name)
        ext = os.path.splitext(self.bboxes_file_name)[1]
        if ext == ".pickle":
            self.mode = "merge"
            index = 0
            with open('/data/veles/tmp/empty_images.txt', 'r') as fin:
                empty = set(fin.readlines())
            self.info("Will load images in interval [%d, %d)", self.min_index,
                      self.max_index)
            with open(self.bboxes_file_name, "rb") as fin:
                while True:
                    try:
                        if self.max_index > 0 and index >= self.max_index:
                            break
                        img = pickle.load(fin)[1]
                        index += 1
                        if index < self.min_index:
                            continue
                        path = img["path"]
                        if path.find(self.only_this_file) < 0:
                            continue
                        if os.path.basename(path) + '\n' not in empty:
                            continue
                        self.bboxes[path] = img
                        size = self.image_size(path)
                        bboxes = img["bbxs"]
                        for bbox in ImagenetForwardLoaderBbox.HARDCODED_BBOXES:
                            x = numpy.round(bbox[0] * size[0])
                            y = numpy.round(bbox[1] * size[1])
                            width = numpy.round(bbox[2] * size[0])
                            height = numpy.round(bbox[3] * size[1])
                            bboxes.append({"x": x, "y": y,
                                           "width": width, "height": height})
                        self.total += len(bboxes)
                    except EOFError:
                        break
            self.info("Loaded %d images", len(self.bboxes))
        elif ext == ".json":
            self.mode = "final"
            with open(self.bboxes_file_name, "r") as fin:
                self.bboxes = {val["path"]: val
                               for val in json.load(fin).values()}
            self.total = sum((len(val["bbxs"])
                              for val in self.bboxes.values()))
        else:
            raise error.BadFormatError()
        self.info("Successfully loaded")
        self.total *= 2  # flip
        self.total *= self.angles
        self.info("Total %d shots", self.total)
        if self.total == 0:
            return
        self._progress = ProgressBar(maxval=self.total, term_width=40,
                                     widgets=['Progress: ', Percentage(), ' ',
                                              Bar()])
        self._progress.start()
        self.bbox_iter = [iter(sorted(self.bboxes.items())), None]
        self._next_image()
        self._initial_state = True

    def initialize(self, device, **kwargs):
        super(ImagenetForwardLoaderBbox, self).initialize(
            device=device, **kwargs)
        shape = self.entry_shape
        self.max_minibatch_size = kwargs.get("minibatch_size", shape[0])
        self.aperture = shape[1]
        self.channels = shape[-1]
        self.add_sobel = self.channels == 4
        self._load_bboxes()

        self.minibatch_data.mem = numpy.zeros(
            (self.max_minibatch_size, self.aperture ** 2 * self.channels),
            dtype=numpy.uint8).reshape(shape)

        self.minibatch_bboxes = [None] * self.max_minibatch_size
        self._last_info_time = time.time()

        self.minibatch_images.extend([""] * self.max_minibatch_size)
        self.minibatch_data.initialize(device)

        if device is not None:
            ImagenetForwardLoaderBbox.ocl_init(self, device)

    def ocl_init(self, device):
        pass

    def reset(self):
        self.total = 0
        self._calc_angles()
        self._load_bboxes()
        self.ended <<= False
        self._last_info_time = time.time()
        self._progress_prevval = 0

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

    def _transform_image(self, img, angle, scale, flip):
        """Transform original image by rotating and scaling, and also adding
        alpha- and Sobel-channels (optionally).

        Args:
            image: 3D numpy array (shape=(height, width, colors_num);
                dtype=numpy.uint8)
            angle: rotation angle in radians
            scale: scaling parameter (1.0 means no scaling)
            flip: if True, flip the image.
            self.add_sobel: flag specifying adding Sobel channel
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
        assert len(img.shape) == 3, "Bad image shape"

        orig_shape = img.shape
        height, width = orig_shape[:2]
        colors_num = orig_shape[2] if len(orig_shape) > 2 else 1

        # calculate optimal output image size
        bbox, _ = self._transform_shape(orig_shape, angle, scale)
        out_width, out_height = (int(numpy.round(numpy.max(bbox[:, i])))
                                 for i in (0, 1))
        tmp_width, tmp_height = max(out_width, width), max(out_height, height)

        # calculate bounding box after rotation and scaling
        center = (tmp_width // 2, tmp_height // 2)
        rot_matrix = cv2.getRotationMatrix2D(
            center, angle * 180 / math.pi, scale)

        # fill RGB color values to expanded image and add alpha-channel
        out_shape = (tmp_height, tmp_width,
                     colors_num + (2 if self.add_sobel else 1))
        out_img = numpy.zeros(shape=out_shape, dtype=numpy.uint8)
        offset_x = max(0, (tmp_width - width) // 2)
        offset_y = max(0, (tmp_height - height) // 2)
        out_img[offset_y:(height + offset_y), offset_x:(width + offset_x),
                :colors_num] = img
        # set alpha channel to 1 as default value
        out_img[offset_y:(height + offset_y), offset_x:(width + offset_x),
                colors_num] = 1

        # rotation and scaling for out_img
        out_img[:, :, :(colors_num + 1)] = cv2.warpAffine(
            out_img[:, :, :(colors_num + 1)], rot_matrix,
            tuple(reversed(out_img.shape[:2])))
        if flip:
            out_img[:, :, :(colors_num + 1)] = cv2.flip(
                out_img[:, :, :(colors_num + 1)], 1)

        # add S-channels to RGBA image (if necessary)
        if self.add_sobel:
            s_img = numpy.zeros((tmp_height, tmp_width), dtype=numpy.uint8)
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) \
                if colors_num > 1 else img

            # get results of Sobel filtration in x- and y-directions
            sobel_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=5)

            # join sobel_x and sobel_y and clip to uint8
            sobel = numpy.sqrt(numpy.square(sobel_x) + numpy.square(sobel_y))
            sobel -= sobel.min()
            max_val = sobel.max()
            if max_val:
                sobel *= 255.0 / max_val
            sobel = numpy.clip(sobel, 0, 255).astype(numpy.uint8)

            s_img[offset_y:(height + offset_y),
                  offset_x:(width + offset_x)] = sobel

            # rotation and scaling for Sobel-channel
            s_img = cv2.warpAffine(s_img, rot_matrix,
                                   tuple(reversed(s_img.shape[:2])))
            if flip:
                cv2.flip(s_img, 1, s_img)
            out_img[:, :, -1] = s_img

        # crop the result in case of scale < 1
        if out_width < width or out_height < height:
            offset_x = max(0, (tmp_width - out_width) // 2)
            offset_y = max(0, (tmp_height - out_height) // 2)
            out_img = out_img[offset_y:(self.aperture + offset_y),
                              offset_x:(self.aperture + offset_x), :]
        return out_img, bbox

    def bbox_is_small(self, bbox):
        width, height = bbox['width'], bbox['height']
        area = width * height
        imsize = self.current_image_size
        if area < max(self.raw_bboxes_min_area,
                      imsize[0] * imsize[1] * self.raw_bboxes_min_area_ratio):
            return True
        if min(width, height) < max(self.raw_bboxes_min_size,
                                    numpy.min(imsize) *
                                    self.raw_bboxes_min_size_ratio):
            return True
        return False

    def _next_state(self):
        if self._initial_state:
            self._initial_state = False
            return self._state
        angle, flip = self._state
        angle += self.angle_step
        if angle > self.max_angle + 0.0001:
            if flip:
                self._state = (self.min_angle, False)
                raise StopIteration()
            self._state = (self.min_angle, True)
        else:
            self._state = (angle, flip)
        return self._state

    def _next_image(self):
        next_img = next(self.bbox_iter[0])
        self.bbox_iter[1] = iter(next_img[1]['bbxs'])
        self.current_image = next_img[0]
        self._current_image_data = self.decode_image(next_img[1]["path"])
        self._current_bbox = self._next_bbox()

    def _next_bbox(self):
        while True:
            try:
                bbox = next(self.bbox_iter[1])
            except StopIteration:
                self._next_image()
                return self._current_bbox
            if not self.bbox_is_small(bbox):
                break
            else:
                self._progress.update(self._progress.currval + 2 * self.angles)
        return bbox

    def _get_bbox_data(self, bbox, angle, flip):
        x, y, width, height = (bbox[p] for p in ('x', 'y', 'width', 'height'))
        xmin = x - width / 2
        ymin = y - height / 2
        xmax = xmin + width
        ymax = ymin + height

        # Crop the image to supplied bbox
        cropped = self.crop_image(self._current_image_data,
                                  (xmin, ymin, xmax, ymax))
        points = numpy.array(((xmin, ymin), (xmax, ymin),
                              (xmax, ymax), (xmin, ymax)))

        # Calculate the scale so that the rotated part takes exactly
        # the aperture square
        rotated_bbox = self._transform_bbox(points, angle, 1.0)
        xmin, ymin = (numpy.min(rotated_bbox[:, i]) for i in (0, 1))
        xmax, ymax = (numpy.max(rotated_bbox[:, i]) for i in (0, 1))
        max_size = max(xmax - xmin, ymax - ymin)
        scale = self.aperture / max_size

        # Rotate the cropped part, scaling to aperture at once and possibly
        # flipping
        sample = self._transform_image(cropped, angle, scale, flip)[0]

        # Last step is to alpha blend with mean image
        try:
            assert sample.shape[0] <= self.aperture and \
                sample.shape[1] <= self.aperture
        except AssertionError:
            self.warning("bbox data overflow: %d %d", *sample.shape[:2])
            sample = sample[:self.aperture, :self.aperture, :]

        lcind = -2 if self.add_sobel else -1
        height, width = sample.shape[:2]
        xoff = (self.aperture - width) // 2
        yoff = (self.aperture - height) // 2
        final = self.mean.mem.copy()
        final[yoff:(yoff + height), xoff:(xoff + width), :] *= \
            (1 - sample[:, :, lcind])[..., None]
        final[yoff:(yoff + height), xoff:(xoff + width),
              :-1 if self.add_sobel else final.shape[-1]] += \
            sample[:, :, :lcind] * sample[:, :, lcind][..., None]
        if self.add_sobel:
            final[yoff:(yoff + height), xoff:(xoff + width), -1] += \
                sample[:, :, -1] * sample[:, :, lcind]
        return final

    def ocl_run(self):
        self.cpu_run()

    def cpu_run(self):
        if self.ended:
            raise NoMoreJobs()
        self.minibatch_data.map_invalidate()
        bbox = self._current_bbox
        now = time.time()
        if now - self._last_info_time > 60:
            self.info(
                "Processed %d / %d (%d%%), took %.1f sec, "
                "will complete in %s",
                self._progress.currval, self._progress.maxval,
                self._progress.percent, now - self._last_info_time,
                datetime.timedelta(seconds=(now - self._last_info_time) / (
                    self._progress.currval - self._progress_prevval) * (
                    self._progress.maxval - self._progress.currval)))
            self._progress_prevval = self._progress.currval
            self._last_info_time = now

        for index in range(self.max_minibatch_size):
            try:
                angle, flip = self._next_state()
            except StopIteration:
                try:
                    self._current_bbox = bbox = self._next_bbox()
                except StopIteration:
                    self.minibatch_size = index
                    self._current_image_data = None
                    self.ended <<= True
                    reactor.callFromThread(self._progress.finish)
                    return
                angle, flip = self._state
            self._progress.inc()
            self.minibatch_data[index] = self._get_bbox_data(bbox, angle, flip)
            self.minibatch_bboxes[index] = (bbox, angle, flip)
            self.minibatch_images[index] = (self.current_image,
                                            self.current_image_size)
        self.minibatch_size = self.max_minibatch_size
