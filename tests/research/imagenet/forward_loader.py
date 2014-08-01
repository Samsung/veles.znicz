"""
Created on Jul 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import cv2
import math
import numpy
from zope.interface import implementer

from veles import OpenCLUnit
import veles.formats as formats
from veles.mutable import Bool
from veles.pickle2 import pickle
from veles.znicz.tests.research.imagenet.processor import Processor
from veles.opencl_units import IOpenCLUnit


@implementer(IOpenCLUnit)
class ImagenetForwardLoaderBbox(OpenCLUnit, Processor):
    """
    Imagenet loader for the first processing stage.

    Defines:
        image_ended          Bool which signals when the current image ends
        ended                Bool which signals when dataset is voer
        minibatch_data       actual data to apply forward propagation
        minibatch_bboxes     corresponding bboxes
        minibatch_size       minibatch size
        current_image        current image file name (dict key)
    """

    def __init__(self, workflow, bboxes_file_name, matrices_pickle, **kwargs):
        kwargs["view_group"] = "LOADER"
        super(ImagenetForwardLoaderBbox, self).__init__(workflow, **kwargs)
        self.bboxes_file_name = bboxes_file_name
        self.aperture = kwargs.get("aperture", 216)
        self.channels = kwargs.get("channels", 4)
        self.max_minibatch_size = kwargs.get('minibatch_size', 128)
        self.minibatch_data = formats.Vector()
        self.minibatch_size = 0
        self.minibatch_bboxes = 0
        self.add_sobel = self.channels == 4
        self.angle_step = kwargs.get("angle_step", numpy.pi / 6)
        self.max_angle = kwargs.get("max_angle", numpy.pi)
        self.ended = Bool()
        self.image_ended = Bool()
        self._state = (0.0, False)  # angle, flip
        self.current_image = ""  # image file name == pickled dict's key
        self._current_image_data = None
        self._next_image_bbox = None  # used when another image appears
        self._mean = None
        self._mean_file_name = matrices_pickle
        # self.demand("entry")  # first forward unit

    def init_unpickled(self):
        super(ImagenetForwardLoaderBbox, self).init_unpickled()

    def initialize(self, device, **kwargs):
        super(ImagenetForwardLoaderBbox, self).initialize(
            device=device, **kwargs)
        self.info("Loading bboxes from %s...", self.bboxes_file_name)
        with open(self.bboxes_file_name, "rb") as fin:
            self.bboxes = pickle.load(fin)
        self.info("Successfully loaded")
        self.info("Loading mean image...")
        with open(self.matrices_filename, "rb") as fin:
            self._mean, _ = pickle.load(fin)
        self.info("Successfully loaded")
        self.bbox_iter = [iter(self.bboxes.items()), None]
        self._next_image()

        self.minibatch_data.mem = numpy.zeros(
            (self.max_minibatch_size, self.aperture ** 2 * self.channels),
            dtype=numpy.uint8)

        self.minibatch_bboxes = numpy.zeros(
            (self.max_minibatch_size, 4, 2), dtype=numpy.uint16)

        self.minibatch_images = [] * self.max_minibatch_size

        if device is None:
            return

        self.minibatch_data.initialize(device)

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
        out_width, out_height = (int(numpy.max(bbox[:, i])) for i in (0, 1))
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
        # set alpha channel to 255 as default value
        out_img[offset_y:(height + offset_y), offset_x:(width + offset_x),
                colors_num] = 255

        # rotation and scaling for out_img
        out_img[:, :, :(colors_num + 1)] = cv2.warpAffine(
            out_img[:, :, :(colors_num + 1)], rot_matrix,
            tuple(reversed(out_img.shape[:2])))
        if flip:
            out_img[:, :, :(colors_num + 1)] = cv2.flip(
                out_img[:, :, :(colors_num + 1)], flipCode=1)

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
                cv2.flip(s_img, s_img, flipCode=1)
            out_img[:, :, -1] = s_img

        # crop the result in case of scale < 1
        if out_width < width or tmp_height < height:
            offset_x = max(0, (tmp_width - out_width) // 2)
            offset_y = max(0, (tmp_height - out_height) // 2)
            out_img = out_img[offset_y:(out_height + offset_y),
                              offset_x:(out_width + offset_x), :]
        return out_img, bbox

    def _next_state(self):
        angle, flip = self._state
        angle += self.angle_step
        if angle > self.max_angle:
            if flip:
                self._state = (0.0, False)
                raise StopIteration()
            self._state = (0.0, True)
        else:
            self._state = (angle, flip)
        return self._state

    def _next_image(self):
        next_img = next(self.bbox_iter[0])
        self.bbox_iter[1] = iter(next_img[1]['bbxs'])
        self.current_image = next_img[0]
        self._current_image_data = self.decode_image(next_img[1]["path"])
        self.image_ended <<= True

    def _next_bbox(self):
        try:
            bbox = next(self.bbox_iter[1])
        except StopIteration:
            self._next_image()
            return self._next_bbox()
        return bbox

    def _get_bbox_data(self, bbox, angle, flip):
        xmin, ymin, xmax, ymax = (bbox['xmin'], bbox['ymin'],
                                  bbox['xmax'], bbox['ymax'])
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
        sample = self._transform_image(cropped, angle, scale, flip)

        # Last step is to alpha blend with mean image
        assert sample.shape[0] == sample.shape[1] == self.aperture
        lcind = -2 if self.add_sobel else -1
        mean = self._mean * (255 - sample[:, :, lcind])[..., None]
        final = numpy.empty((self.aperture, self.aperture,
                             sample.shape[-1] - 1), dtype=sample.dtype)
        final[:, :, :-1] = sample[:, :, :lcind] + mean[:, :, :-1]
        if self.add_sobel:
            final[:, :, -1] = sample[:, :, -1] + mean[:, :, -1]
        return final

    def ocl_run(self):
        self.cpu_run()

    def cpu_run(self):
        self.minibatch_data.map_invalidate()

        for index in range(self.max_minibatch_size):
            self.image_ended <<= False
            self.minibatch_size = index
            if self._next_image_bbox is not None:
                bbox, self._next_image_bbox = self._next_image_bbox, None
            else:
                try:
                    angle, flip = self._next_state()
                except StopIteration:
                    try:
                        bbox = self._next_bbox()
                    except StopIteration:
                        self.ended <<= True
                        return
            if self.image_ended:
                self._next_image_bbox = bbox
            else:
                self.minibatch_data[index] = \
                    self._get_bbox_data(bbox, angle, flip)
                self.minibatch_bboxes[index] = bbox
        self.minibatch_size = self.max_minibatch_size
