"""
Created on Jul 11, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import cv2
import jpeg4py
import numpy
from PIL import Image

from veles import Logger


class Processor(Logger):
    """
    Accumulates common routines to work with Imagenet data.
    """

    def decode_image(self, file_name):
        try:
            data = jpeg4py.JPEG(file_name).decode()
        except jpeg4py.JPEGRuntimeError as e:
            try:
                data = numpy.array(Image.open(file_name).convert("RGB"))
                self.warning("Falling back to PIL with file %s: %s",
                             file_name, repr(e))
            except Exception as e:
                try:
                    data = cv2.imread(file_name)
                    self.warning("Falling back to OpenCV with file %s: %s",
                                 file_name, repr(e))
                except:
                    self.exception("Failed to decode %s", file_name)
                    raise
        return data

    def crop_image(self, img, bbox):
        xmin, ymin, xmax, ymax = bbox
        return img[ymin:ymax, xmin:xmax]
