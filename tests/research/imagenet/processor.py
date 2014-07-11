"""
Created on Jul 11, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


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
            except:
                self.exception("Failed to decode %s", file_name)
                raise
        return data
