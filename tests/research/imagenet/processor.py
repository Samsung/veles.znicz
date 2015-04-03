"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jul 11, 2014

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


import cv2
import jpeg4py
import numpy
from PIL import Image

from veles.logger import Logger


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

    def image_size(self, path):
        return Image.open(path).size
