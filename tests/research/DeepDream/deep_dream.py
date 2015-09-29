"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on August 27, 2015

DeepDream workflow. This workflow uses https://github.com/google/deepdream
code, which was released under Apache 2.0 License

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
import numpy
import pickle
from PIL import Image
import os
from scipy import ndimage, misc

from veles import prng
from veles.config import root
from veles.memory import Array
from veles.snapshotter import SnapshotterToFile
from veles.znicz.nn_units import NNWorkflow


class DeepDreamWorkflow(NNWorkflow):
    def __init__(self, workflow, **kwargs):
        super(DeepDreamWorkflow, self).__init__(workflow, **kwargs)
        self.mean_base = Array()
        self.snapshot_name = kwargs["snapshot_name"]
        self.train_wf = SnapshotterToFile.import_(self.snapshot_name)
        self.train_wf.stopped = True
        self.train_wf.workflow = self
        self.train_wf.gds[0].need_err_input = True
        self.train_wf.link_from(self.start_point)
        self.end_point.link_from(self.train_wf)
        self.rect = kwargs.get("rect", (256, 256))
        self.crop_size_sx = kwargs.get("crop_size_sx", 227)
        self.crop_size_sy = kwargs.get("crop_size_sy", 227)
        self.matrixes_filename = kwargs.get("matrixes_filename", None)
        self.path_to_output = kwargs["path_to_output"]
        self.path_to_input = kwargs["path_to_input"]
        self.filter_names = ("JPEG", "png", "jpg")
        self.visualize = kwargs.get("visualize", True)
        self.path_to_temp_output = kwargs.get(
            "path_to_temp_output", self.path_to_output)
        if self.visualize and not os.access(self.tokens_file, os.R_OK):
            raise OSError(
                "Path to directory %s does not exist. Set 'path_to_"
                "temp_output' or change 'visualize' to False" %
                self.path_to_temp_output)
        if self.matrixes_filename is not None:
            with open(self.matrixes_filename, "rb") as fin:
                matrixes = pickle.load(fin)
            self.mean_base.mem = matrixes[0]
            self.mean = self.cut_out(self.mean_base.mem)
        else:
            self.mean = None
        self.with_affine_transform = kwargs.get("with_affine_transform", False)
        self.number_of_iterations = kwargs.get("number_of_iterations", 10)
        self.number_of_octaves = kwargs.get("number_of_octaves", 4)
        self.octave_scale = kwargs.get("octave_scale", 1.4)
        self.clip = kwargs.get("clip", True)
        self.end_unit_name = kwargs.get("end_unit_name", "fc_linear7_forward")
        self.end_gd = kwargs.get("end_gd", "gd_fc_linear7_backward")
        # TODO: get end_gd name from end_unit_name
        self.step_size = kwargs.get("step_size", 1.5)
        self.jitter = kwargs.get("jitter", 32)
        self.affine_scale = kwargs.get("affine_scale", 0.05)
        self.affine_iterations = kwargs.get("affine_iterations", 100)
        if self.affine_scale is None and self.with_affine_transform is True:
            raise OSError("For with_affine_transform set 'affine_scale'")
        if (self.affine_iterations is None and
                self.with_affine_transform is True):
            raise OSError("For with_affine_transform set 'affine_iterations'")
        self.end_point.link_from(self.start_point)

    def make_step(self, image, step_size, end_unit_name, end_gd, jitter, clip):
        """Basic gradient ascent step.
        """
        rand = prng.get()

        # apply jitter shift
        ox, oy = rand.randint(-jitter, jitter + 1, 2)
        image = numpy.roll(numpy.roll(image, ox, -2), oy, 0)

        self.train_wf.forwards[0].input.map_invalidate()
        self.train_wf.forwards[0].input.mem[:] = image
        self.train_wf.forward(end_unit=end_unit_name)

        self.train_wf[end_gd].err_output.map_invalidate()
        self.train_wf[end_gd].output.map_read()
        self.train_wf[end_gd].err_output.mem[:] = \
            self.train_wf[end_gd].output.mem[:]

        self.train_wf.backward(start_unit=end_gd)

        self.train_wf.gds[0].err_input.map_read()
        g = self.train_wf.gds[0].err_input.mem

        self.train_wf.forwards[0].input.map_invalidate()
        self.train_wf.forwards[0].input.mem[:] += step_size / numpy.abs(
            g).mean() * g

        self.train_wf.forwards[0].input.map_read()
        image = self.train_wf.forwards[0].input.mem[0]

        # unshift image
        image = numpy.roll(numpy.roll(image, -ox, -2), -oy, 0)

        self.train_wf.forwards[0].input.map_invalidate()
        self.train_wf.forwards[0].input.mem[0] = image

        if clip:
            self.train_wf.forwards[0].input.map_read()
            bias = self.mean
            images = self.train_wf.forwards[0].input.mem
            images[:] = numpy.clip(images, -bias, 255 - bias)
            image = images[0]

        return image

    def preprocess(self, image, mean):
        image = cv2.resize(
            image, (self.crop_size_sx, self.crop_size_sy),
            interpolation=cv2.INTER_LANCZOS4)
        if mean is not None:
            image -= mean
        return image

    @staticmethod
    def deprocess(image, mean, w, h):
        if mean is not None:
            image += mean
        image = cv2.resize(
            image, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return image

    @staticmethod
    def resize_to_original(img, original_h, original_w):
        sample = cv2.resize(
            img, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
        return sample

    def cut_out(self, sample):
        h_off = (sample.shape[0] - self.crop_size_sy) / 2
        w_off = (sample.shape[1] - self.crop_size_sx) / 2
        sample = sample[
            h_off:h_off + self.crop_size_sy,
            w_off:w_off + self.crop_size_sx, :3]
        return sample

    def deep_deep_dream(self, img, file_name):
        frame = img
        h, w = frame.shape[:2]
        s = self.affine_scale
        for frame_i in range(self.affine_iterations):
            frame = self.deep_dream(
                frame, file_name, iter_n=self.number_of_iterations,
                octave_n=self.number_of_octaves,
                octave_scale=self.octave_scale, clip=self.clip)
            file_path_output = os.path.join(
                self.path_to_output,
                "frames/%s_%04d.jpg" % (file_name, frame_i))
            misc.imsave(file_path_output, frame)
            frame = ndimage.affine_transform(
                frame, [1 - s, 1 - s, 1], [h * s / 2, w * s / 2, 0], order=1)
        file_path_output = os.path.join(
            self.path_to_output,
            "%s_final_output.png" % file_name)
        misc.imsave(file_path_output, frame)
        self.info("Saved output picture to %s" % file_path_output)

    def deep_dream(self, img, file_name, iter_n, octave_n, octave_scale, clip):
        base_img = numpy.zeros(
            [img.shape[0], img.shape[1], 3], dtype=img.dtype)
        base_img[:, :, 0:3] = img[:, :, 0:3]

        original_h = base_img.shape[0]
        original_w = base_img.shape[1]

        octaves = [base_img]

        for i in range(octave_n - 1):
            scale = 1.0 / (octave_scale ** (octave_n - i))
            octaves.append(
                ndimage.zoom(base_img, (scale, scale, 1), order=1))

        detail = numpy.zeros_like(octaves[-1])

        self.train_wf.forwards[0].input.map_read()
        image = self.train_wf.forwards[0].input.mem[0]

        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[:2]
            if octave > 0:
                h1, w1 = detail.shape[:2]
                detail = ndimage.zoom(detail, (1.0 * h / h1, 1.0 * w / w1, 1),
                                      order=1)

            image = octave_base + detail
            image = self.preprocess(image, self.mean)

            for i in range(iter_n):
                image = self.make_step(
                    image, step_size=self.step_size,
                    end_unit_name=self.end_unit_name,
                    end_gd=self.end_gd, jitter=self.jitter, clip=clip)

                vis = self.resize_to_original(image, original_h, original_w)
                if not clip:
                    vis = vis * (255.0 / numpy.percentile(vis, 99.98))

                if self.visualize:
                    misc.imsave(
                        os.path.join(
                            self.path_to_temp_output,
                            "output_%s_octave_%s_iteration.png" % (octave, i)),
                        vis)

            image = self.deprocess(image, self.mean, w, h)
            detail = image - octave_base

        img = self.resize_to_original(image, original_h, original_w)
        file_path_output = os.path.join(
            self.path_to_output,
            "%s_output.png" % file_name)
        misc.imsave(file_path_output, img)
        self.info("Saved output picture to %s" % file_path_output)
        return img

    def run(self):
        for root_path, _tmp, files in os.walk(
                self.path_to_input, followlinks=True):
            for file_name in files:
                for filter_name in self.filter_names:
                    if not file_name.endswith(filter_name):
                        continue
                    file_path = os.path.join(root_path, file_name)
                    self.info("Will run picture %s" % file_path)
                    img = numpy.float32(Image.open(file_path))
                    file_name = file_name[:file_name.rfind(".")]
                    if self.with_affine_transform is True:
                        self.deep_deep_dream(
                            img, file_name)
                    else:
                        self.deep_dream(
                            img, file_name, iter_n=self.number_of_iterations,
                            octave_n=self.number_of_octaves,
                            octave_scale=self.octave_scale, clip=self.clip)
        for unit in self:
            unit.stopped = False
        super(DeepDreamWorkflow, self).run()


def run(load, main):
    load(DeepDreamWorkflow,
         path_to_input=root.deep_dream.path_to_input,
         path_to_output=root.deep_dream.path_to_output,
         path_to_temp_output=root.deep_dream.path_to_temp_output,
         snapshot_name=root.deep_dream.snapshot_name,
         matrixes_filename=root.deep_dream.matrixes_filename,
         **root.deep_dream.parameters.__content__)
    main()
