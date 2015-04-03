# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 19, 2014

Unit test for convolutional layer forward propagation, compared to CAFFE data.

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


import numpy
import os
from scipy.signal import correlate2d, convolve2d  # pylint: disable=E0611

from veles.dummy import DummyUnit
from veles.memory import Vector
import veles.znicz.all2all as all2all
import veles.znicz.conv as conv
import veles.znicz.evaluator as evaluator
import veles.znicz.gd_conv as gd_conv
import veles.znicz.gd as gd
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.normalization as normalization
import veles.znicz.pooling as pooling
from veles.znicz.tests.functional import StandardTest


class CaffeTestBase(StandardTest):
    def _read_array(self, array_name, lines, shape=None):
        """
        Reads a pic array from from export file, splitted to lines.
        NB: last line should be empty

        Arguments:
            array_name(str): name of array to read
        lines(array): lines of file to read from
        shape(tuple): shape=(n_pics, height, width, n_chans), must be given if
            not set in file.

        Returns:
            :class:`numpy.ndarray`
        """

        cur_line = None
        for i, line in enumerate(lines):
            line = line.replace("\n", "")
            nibbles = line.split("\t")
            if nibbles[0] == array_name:
                if len(nibbles) >= 5:  # shape is set in file
                    dimensions = {}
                    for nibble in nibbles[1:]:
                        [nibble_name, nibble_val] = nibble.split(":")
                        dimensions[nibble_name] = int(nibble_val)
                    n_pics = dimensions["num"]
                    height = dimensions["height"]
                    width = dimensions["width"]
                    n_chans = dimensions["channels"]
                    if shape is not None:
                        assert shape == (n_pics, height, width, n_chans)
                else:  # shape is set externally
                    assert len(shape) == 4
                    n_pics, height, width, n_chans = shape

                out_array = numpy.zeros((n_pics, height, width, n_chans),
                                        numpy.float64)
                cur_line = i + 1
                break

        assert cur_line is not None
        assert cur_line < len(lines)

        for cur_pic in range(n_pics):
            nibbles = lines[cur_line].split(":")
            assert nibbles[0] == "num"
            assert int(nibbles[1]) == cur_pic
            cur_line += 1

            for cur_chan in range(n_chans):
                nibbles = lines[cur_line].split(":")
                assert nibbles[0] == "channels"
                assert int(nibbles[1]) == cur_chan
                cur_line += 1

                for i in range(height):
                    data = [float(x) for x in lines[cur_line].split("\t")]
                    cur_line += 1

                    for j in range(width):
                        out_array[cur_pic, i, j, cur_chan] = data[j]
        return out_array

    def _read_lines(self, data_filename):
        """
        Returns all lines from a file maned `data_filename`.
        File is searched in ``self.data_dir_path``.

        Arguments:
            data_filename(str): name to file with pooling data,
                exported from CAFFE (searched in ``self.data_dir_path``)

        Returns:
            list: list of all lines read
        """
        full_path = os.path.join(self.data_dir_path, data_filename)
        return self._read_lines_by_abspath(full_path)

    def _read_lines_by_abspath(self, full_path):
        with open(full_path, 'r') as in_file:
            return in_file.readlines()


class TestConvCaffe(CaffeTestBase):
    def test_caffe_conv(self, data_filename="conv.txt"):
        """
        Compare CAFFE conv layer fwd prop with Veles conv layer.

        Args:
            data_filename(str): name of file with pooling data
                (relative from data dir)
        """
        lines = self._read_lines(data_filename)

        kernel_size = 5
        padding_size = 2

        bottom = self._read_array("bottom", lines=lines, shape=(2, 32, 32, 3))
        weights = self._read_array("weights", lines=lines, shape=(2, 5, 5, 3))
        top = self._read_array("top", lines=lines, shape=(2, 32, 32, 2))

        fwd_conv = conv.Conv(self.parent, kx=kernel_size, ky=kernel_size,
                             padding=(padding_size, padding_size,
                                      padding_size, padding_size),
                             sliding=(1, 1),
                             n_kernels=2)

        fwd_conv.input = Vector()
        fwd_conv.input.mem = bottom

        fwd_conv.initialize(self.device)
        fwd_conv.weights.map_invalidate()
        fwd_conv.weights.mem[:] = weights.reshape(2, 75)[:]
        fwd_conv.bias.map_invalidate()
        fwd_conv.bias.mem[:] = 0
        fwd_conv.run()

        self.info("Veles vs CAFFE data:")
        fwd_conv.output.map_read()

        self.info("Veles top shape:" + str(fwd_conv.output.mem.shape))
        delta_with_veles = fwd_conv.output.mem - top

        self.info("CONV: diff with Veles: %.2f%%" % (
            100. * numpy.sum(numpy.abs(delta_with_veles)) /
            numpy.sum(numpy.abs(fwd_conv.output.mem)),))

        self.info("COMPARED TO HANDMADE CORRELATION:")
        scipy_conv_out = numpy.zeros(shape=(2, 32, 32, 2), dtype=numpy.float64)

        for pic in range(2):
            for color_chan in range(3):
                for weight_id in range(2):
                    correlation = correlate2d(
                        bottom[pic, :, :, color_chan],
                        weights[weight_id, :, :, color_chan], mode="same")
                    scipy_conv_out[pic, :, :, weight_id] += correlation

        delta_with_scipy = fwd_conv.output.mem - scipy_conv_out
        self.info("CONV: diff with SciPy: %.2f%%" % (
            100. * numpy.sum(numpy.abs(delta_with_scipy)) /
            numpy.sum(numpy.abs(fwd_conv.output.mem)),))

    def test_caffe_grad_conv(self, data_filename="conv_grad.txt"):
        """
        Compare CAFFE conv layer with Veles conv layer (FwdProp and BackProp).

        Args:
            data_filename(str): name of file with pooling data
                (relative from data dir)
        """
        lines = self._read_lines(data_filename)

        # stride = 1
        bot_size = 32
        top_size = 32
        kernel_size = 5
        padding_size = 2
        n_kernels = 2
        batch_size = 2

        bottom = self._read_array("bottom", lines=lines,
                                  shape=(batch_size, bot_size, bot_size, 3))

        weights = self._read_array("weights", lines=lines,
                                   shape=(n_kernels,
                                          kernel_size,
                                          kernel_size,
                                          3))
        top = self._read_array("top", lines=lines,
                               shape=(batch_size, top_size, top_size, 2))

        top_err = self._read_array("top_diff", lines=lines,
                                   shape=(batch_size,
                                          top_size,
                                          top_size,
                                          2))

        bot_err = self._read_array("bottom_diff", lines=lines,
                                   shape=(batch_size,
                                          bot_size,
                                          bot_size,
                                          3))

        fwd_conv = conv.Conv(self.parent, kx=kernel_size, ky=kernel_size,
                             padding=(padding_size, padding_size,
                                      padding_size, padding_size),
                             sliding=(1, 1), n_kernels=n_kernels)

        fwd_conv.input = Vector(bottom)

        fwd_conv.initialize(self.device)
        fwd_conv.weights.map_invalidate()
        fwd_conv.weights.mem[:] = weights.reshape(2, 75)[:]
        fwd_conv.bias.map_invalidate()
        fwd_conv.bias.mem[:] = 0
        fwd_conv.run()

        self.info("Veles vs CAFFE data:")
        fwd_conv.output.map_read()

        self.info("Veles top shape:" + str(fwd_conv.output.mem.shape))
        delta_with_veles = fwd_conv.output.mem - top

        self.info("CONV: diff with CAFFE: %.2f%%" % (
            100. * numpy.sum(numpy.abs(delta_with_veles)) /
            numpy.sum(numpy.abs(fwd_conv.output.mem)),))

        back_conv = gd_conv.GradientDescentConv(self.parent)

        back_conv.input = Vector(bottom)

        back_conv.output = Vector(top)

        back_conv.err_output = Vector(top_err)

        back_conv.weights = Vector()
        back_conv.weights.mem = fwd_conv.weights.mem

        back_conv.bias = Vector(fwd_conv.bias.mem)

        back_conv.batch_size = 2

        back_conv.link_conv_attrs(fwd_conv)

        back_conv.initialize(self.device)

        back_conv.run()

        back_conv.err_input.map_read()

        # BACKPROP: difference with CAFFE export
        back_delta = back_conv.err_input.mem - bot_err

        self.info("GDCONV: diff with CAFFE: %.3f%%" %
                  (100. * numpy.sum(numpy.fabs(back_delta)) /
                   numpy.sum(numpy.fabs(back_conv.err_input.mem)),))

        # perform manual GD CONV
        manual_bot_err = numpy.zeros(shape=(2, bot_size, bot_size, 3),
                                     dtype=numpy.float64)
        for pic in range(batch_size):
            for color_chan in range(3):
                for weight_id in range(n_kernels):
                    conv_result = convolve2d(
                        top_err[pic, :, :, weight_id],
                        weights[weight_id, :, :, color_chan], mode="same")
                    manual_bot_err[pic, :, :, color_chan] += conv_result

        self.info("Manual GDCONV: diff with CAFFE: %.3f%%" % (
            100. * numpy.sum(numpy.fabs(manual_bot_err - bot_err)) /
            numpy.sum(numpy.fabs(bot_err))))

    def test_caffe_pooling(self, data_filename="pool.txt"):
        """
        Compare CAFFE pooling unit fwd_prop with Veles one

        Args:
            data_filename(str): name to file with pooling data,
                exported from CAFFE (searched in ``self.data_dir_path``)
        """

        # load pooling data from CAFFE dumps
        in_file = open(os.path.join(self.data_dir_path, data_filename), 'r')
        lines = in_file.readlines()
        in_file.close()

        # max pooling: 3x3 kernel, 2x2 stride
        kernel_size = 3
        stride = 2

        in_height, in_width = 32, 32
        out_height, out_width = 16, 16

        bottom = self._read_array("bottom", lines=lines,
                                  shape=(2, in_height, in_width, 2))

        # do pooling with VELES
        fwd_pool = pooling.MaxPooling(self.parent, kx=kernel_size,
                                      ky=kernel_size, sliding=(stride, stride),
                                      device=self.device)
        fwd_pool.input = Vector(bottom)
        fwd_pool.input.map_write()

        fwd_pool.initialize(device=self.device)

        fwd_pool.cpu_run()
        fwd_pool.output.map_read()

        # do MANUAL pooling
        manual_pooling_out = numpy.zeros(shape=(2, out_height, out_width, 2),
                                         dtype=numpy.float64)
        for pic in range(2):
            for chan in range(2):
                for i_out in range(out_height):
                    for j_out in range(out_width):
                        min_i = i_out * 2
                        max_i = i_out * 2 + kernel_size - 1
                        min_j = j_out * 2
                        max_j = j_out * 2 + kernel_size - 1

                        zone = bottom[pic, min_i: max_i + 1, min_j:
                                      max_j + 1, chan]

                        manual_pooling_out[pic, i_out, j_out,
                                           chan] = numpy.max((zone))

    def test_caffe_grad_pooling(self, data_filename="pool_grad.txt"):
        """
        Compare CAFFE pooling unit with Veles ones (fwd and back propagations)

        Args:
            data_filename(str): name of file with pooling data
                (relative from data dir)
        """
        bot_size = 32
        top_size = 16
        kernel_size = 3
        stride = 2
        n_chans = 2
        n_pics = 2

        lines = self._read_lines(data_filename)
        lines = [line.replace("\t\n", "").replace("\n", "") for line in lines]

        bottom = self._read_array("bottom", lines,
                                  shape=(n_pics, bot_size, bot_size, n_chans))
        top = self._read_array("top", lines,
                               shape=(n_pics, top_size, top_size, n_chans))
        bot_err = self._read_array("bottom_diff", lines,
                                   shape=(n_pics, bot_size, bot_size, n_chans))
        top_err = self._read_array("top_diff", lines,
                                   shape=(n_pics, top_size, top_size, n_chans))

        # FORWARD PROP
        fwd_pool = pooling.MaxPooling(self.parent, kx=kernel_size,
                                      ky=kernel_size, sliding=(stride, stride))
        fwd_pool.input = Vector(bottom)
        fwd_pool.input.map_write()

        fwd_pool.initialize(device=self.device)

        fwd_pool.cpu_run()
        fwd_pool.output.map_read()

        self.info("FWD POOL: Veles vs CAFFE: %.3f%%" %
                  (100. * (numpy.sum(numpy.abs(fwd_pool.output.mem - top)) /
                           numpy.sum(numpy.abs(top)))))

        # Do MANUAL pooling
        out_height, out_width = top_size, top_size
        manual_pooling_out = numpy.zeros(shape=(2, out_height, out_width, 2),
                                         dtype=numpy.float64)
        for pic in range(2):
            for chan in range(2):
                for i_out in range(out_height):
                    for j_out in range(out_width):
                        min_i = i_out * stride
                        max_i = i_out * stride + kernel_size - 1
                        min_j = j_out * stride
                        max_j = j_out * stride + kernel_size - 1

                        zone = bottom[pic, min_i: max_i + 1,
                                      min_j: max_j + 1, chan]

                        manual_pooling_out[pic, i_out, j_out,
                                           chan] = numpy.max((zone))

        # BACK PROP
        grad_pool = gd_pooling.GDMaxPooling(self.parent, kx=kernel_size,
                                            ky=kernel_size,
                                            sliding=(stride, stride),
                                            device=self.device)
        grad_pool.input = Vector(bottom)
        grad_pool.input.map_write()

        grad_pool.err_output = Vector(top_err)
        grad_pool.err_output.map_write()

        grad_pool.input_offset = fwd_pool.input_offset

        grad_pool.link_pool_attrs(fwd_pool)

        grad_pool.initialize(device=self.device)

        grad_pool.cpu_run()

        grad_pool.err_input.map_read()
        self.info("BACK POOL: Veles vs CAFFE, %.3f%%" % (100 * numpy.sum(
            numpy.abs(grad_pool.err_input.mem - bot_err)) /
            numpy.sum(numpy.abs(bot_err))))

    def test_caffe_grad_normalization(self, data_filename="norm_gd.txt"):
        """
        Tests LRU normalization unit: compares it with CAFFE one.
        Fwd and back props made.

        Args:
            data_filename(str): name of file with pooling data
                (relative from data dir)
        """
        lines = self._read_lines(data_filename)

        size = 16
        n_chans = 2
        n_pics = 2

        max_percent_delta = 2.  # max difference with CAFFE (percents)

        bottom = self._read_array("bottom", lines,
                                  shape=(n_pics, size, size, n_chans))
        top = self._read_array("top", lines,
                               shape=(n_pics, size, size, n_chans))

        bot_err = self._read_array("bottom_diff", lines,
                                   shape=(n_pics, size, size, n_chans))

        top_err = self._read_array("top_diff", lines,
                                   shape=(n_pics, size, size, n_chans))

        fwd_norm = normalization.LRNormalizerForward(self.parent,
                                                     k=1, device=self.device)

        # FWD PROP
        fwd_norm.input = Vector(bottom)

        fwd_norm.initialize(self.device)
        fwd_norm.run()

        fwd_norm.output.map_read()

        fwd_percent_delta = 100. * (
            numpy.sum(numpy.abs(fwd_norm.output.mem - top)) /
            numpy.sum(numpy.abs(top)))

        self.info("FWD NORM DELTA: %.2f%%" % fwd_percent_delta)
        self.assertLess(fwd_percent_delta, max_percent_delta,
                        "Fwd norm differs by %.2f%%" % fwd_percent_delta)

        # BACK PROP
        back_norm = normalization.LRNormalizerBackward(self.parent,
                                                       k=1, device=self.device)
        back_norm.output = Vector(top)

        back_norm.input = Vector(bottom)

        back_norm.err_output = Vector(top_err)

        back_norm.initialize(self.device)
        back_norm.run()

        back_norm.err_input.map_read()

        back_percent_delta = 100. * numpy.sum(
            numpy.abs(back_norm.err_output.mem - bot_err)) / \
            numpy.sum(numpy.abs(bot_err))

        self.info("BACK NORM DELTA: %.2f%%" % back_percent_delta)
        self.assertLess(back_percent_delta, max_percent_delta,
                        "Fwd norm differs by %.2f%%" % (back_percent_delta))

    def test_caffe_relu(self, data_filename="conv_relu.txt"):
        """
        Tests CONV+RELU unit: compares it with CAFFE one.
        Fwd prop only.

        Args:
            data_filename(str): name of file with pooling data
                (relative from data dir)
        """
        lines = self._read_lines(data_filename)

        in_size = 32
        out_size = 32
        n_chans = 3
        n_kernels = 2
        n_pics = 2
        kernel_size = 5
        padding_size = 2

        max_percent_delta = 2.0

        conv_bottom = self._read_array(
            "conv_bottom", lines, shape=(n_pics, in_size, in_size, n_chans))
        conv_top = self._read_array(
            "conv_top", lines, shape=(n_pics, out_size, out_size, n_kernels))
        relu_top_flat = self._read_array(
            "relu_top_flat", lines, shape=(1, 1, conv_top.size, 1)).ravel()
        relu_top = numpy.ndarray(
            shape=(n_pics, out_size, out_size, n_kernels), dtype=numpy.float64)
        cur_pos = 0
        for pic in range(n_pics):
            for kernel in range(n_kernels):
                for i in range(out_size):
                    for j in range(out_size):
                        relu_top[pic, i, j, kernel] = relu_top_flat[cur_pos]
                        cur_pos += 1

        conv_weights = self._read_array("conv_weights", lines, shape=(
            n_kernels, kernel_size, kernel_size, n_chans))

        fwd_conv_relu = conv.ConvStrictRELU(
            self.parent, kx=kernel_size, ky=kernel_size,
            padding=(padding_size, padding_size, padding_size, padding_size),
            sliding=(1, 1), n_kernels=n_kernels, device=self.device)

        fwd_conv_relu.input = Vector(conv_bottom)

        fwd_conv_relu.initialize(self.device)

        fwd_conv_relu.weights.map_invalidate()
        fwd_conv_relu.weights.mem[:] = conv_weights.reshape(2, 75)[:]
        fwd_conv_relu.bias.map_invalidate()
        fwd_conv_relu.bias.mem[:] = 0

        fwd_conv_relu.run()

        fwd_conv_relu.output.map_read()

        percent_delta = 100. * (numpy.sum(numpy.abs(
            fwd_conv_relu.output.mem - relu_top)) /
            numpy.sum(numpy.abs(relu_top)))

        self.info("CONV_RELU: diff with CAFFE %.3f%%" % percent_delta)
        self.assertLess(percent_delta, max_percent_delta,
                        "Fwd ConvRELU differs by %.2f%%" % percent_delta)

        relu_top_manual = numpy.where(numpy.greater(conv_top, 0), conv_top, 0)
        manual_delta = 100. * numpy.sum(
            numpy.abs(relu_top_manual - relu_top)) /\
            (numpy.sum(numpy.abs(relu_top)))
        self.info("SciPy CONV_RELU: diff with CAFFE %.3f%%" % manual_delta)

    def test_caffe_grad_relu(self, data_filename="conv_relu.txt"):
        """
        Tests CONV+RELU unit: compares it with CAFFE one.
        Fwd prop only.

        Args:
            data_filename(str): name of file with pooling data
                (relative from data dir)
        """
        lines = self._read_lines(data_filename)

        in_size = 32
        out_size = 32
        n_chans = 3
        n_kernels = 2
        n_pics = 2
        kernel_size = 5
        padding_size = 2

        max_percent_delta = 2.0

        conv_bottom = self._read_array(
            "conv_bottom", lines, shape=(n_pics, in_size, in_size, n_chans))
        conv_top = self._read_array(
            "conv_top", lines, shape=(n_pics, out_size, out_size, n_kernels))
        relu_top_flat = self._read_array(
            "relu_top_flat", lines, shape=(1, 1, conv_top.size, 1)).ravel()
        relu_top = numpy.ndarray(shape=(n_pics, out_size, out_size, n_kernels),
                                 dtype=numpy.float64)
        cur_pos = 0
        for pic in range(n_pics):
            for kernel in range(n_kernels):
                for i in range(out_size):
                    for j in range(out_size):
                        relu_top[pic, i, j, kernel] = relu_top_flat[cur_pos]
                        cur_pos += 1

        conv_weights = self._read_array("conv_weights", lines, shape=(
            n_kernels, kernel_size, kernel_size, n_chans))

        fwd_conv_relu = conv.ConvStrictRELU(
            self.parent, kx=kernel_size, ky=kernel_size,
            padding=(padding_size, padding_size, padding_size, padding_size),
            sliding=(1, 1), n_kernels=n_kernels, device=self.device)

        fwd_conv_relu.input = Vector(conv_bottom)

        fwd_conv_relu.initialize(self.device)

        fwd_conv_relu.weights.map_invalidate()
        fwd_conv_relu.weights.mem[:] = conv_weights.reshape(2, 75)[:]
        fwd_conv_relu.bias.map_invalidate()
        fwd_conv_relu.bias.mem[:] = 0

        fwd_conv_relu.run()

        fwd_conv_relu.output.map_read()

        percent_delta = 100. * (numpy.sum(numpy.abs(
            fwd_conv_relu.output.mem - relu_top)) /
            numpy.sum(numpy.abs(relu_top)))

        self.info("CONV_RELU: diff with CAFFE %.3f%%" % percent_delta)
        self.assertLess(percent_delta, max_percent_delta,
                        "Fwd ConvRELU differs by %.2f%%" % percent_delta)

        relu_top_manual = numpy.where(numpy.greater(conv_top, 0), conv_top, 0)
        manual_delta = 100. * numpy.sum(numpy.abs(relu_top_manual - relu_top))\
            / numpy.sum(numpy.abs(relu_top))
        self.info("SciPy CONV_RELU: diff with CAFFE %.3f%%" % manual_delta)

    def test_grad_conv_relu(self, data_filename="conv_relu_grad.txt"):
        """
        Tests GDDRELU_CONV unit: compares it with CAFFE one.
        Backward prop only

        Args:
            data_filename(str): name of file with pooling data
                (relative from data dir)
        """
        lines = self._read_lines(data_filename)

        in_size = 32
        out_size = 32
        n_chans = 3
        n_kernels = 2
        n_pics = 2
        kernel_size = 5
        padding = 2

        max_percent_delta = 2.0

        conv_bot_err = self._read_array("conv_bottom_diff", lines,
                                        shape=(n_pics, in_size, in_size,
                                               n_chans))
        conv_bottom = self._read_array(
            "conv_bottom", lines, shape=(n_pics, in_size, in_size, n_chans))

        conv_weights = self._read_array("conv_weights", lines, shape=(
            n_kernels, kernel_size, kernel_size, n_chans))

        conv_weight_delta = self._read_array(
            "conv_weight_delta", lines,
            shape=(n_kernels, kernel_size, kernel_size, n_chans))

        relu_top_err = self._read_array("relu_top_diff", lines, shape=(n_pics,
                                        in_size, in_size, n_kernels))

        relu_top_flat = self._read_array("relu_top_flat", lines, shape=(
            1, 1, relu_top_err.size, 1)).ravel()

        relu_top = numpy.ndarray(shape=(n_pics, out_size, out_size, n_kernels),
                                 dtype=numpy.float64)
        cur_pos = 0
        for pic in range(n_pics):
            for kernel in range(n_kernels):
                for i in range(out_size):
                    for j in range(out_size):
                        relu_top[pic, i, j, kernel] = relu_top_flat[cur_pos]
                        cur_pos += 1

        # Testing back prop
        back_conv_relu = gd_conv.GDStrictRELUConv(self.parent,
                                                  device=self.device,
                                                  learning_rate=1,
                                                  weights_decay=0,
                                                  batch_size=n_pics)

        back_conv_relu.err_output = Vector(relu_top_err)
        back_conv_relu.input = Vector(conv_bottom)

        back_conv_relu.weights = Vector(conv_weights.reshape(2, 75))
        back_conv_relu.bias = Vector(numpy.zeros(shape=n_kernels))

        back_conv_relu.output = Vector(relu_top)

        back_conv_relu.link_conv_attrs(
            DummyUnit(kx=kernel_size, ky=kernel_size, n_kernels=n_kernels,
                      padding=((padding,) * 4), sliding=(1, 1),
                      unpack_size=1, unpack_data=Vector()))

        back_conv_relu.initialize(device=self.device)

        back_conv_relu.weights.map_invalidate()
        back_conv_relu.weights.mem[:] = conv_weights.reshape(2, 75)[:]
        back_conv_relu.bias.map_invalidate()
        back_conv_relu.bias.mem[:] = 0

        back_conv_relu.cpu_run()

        back_conv_relu.err_input.map_read()
        result = back_conv_relu.err_input.mem

        percent_delta = 100. * (numpy.sum(numpy.abs(result - conv_bot_err))
                                / numpy.sum(numpy.abs(result)))
        self.info("GD_CONV_RELU: diff with CAFFE %.3f%%" % percent_delta)
        self.assertLess(percent_delta, max_percent_delta,
                        "Fwd GD_ConvRELU differs by %.2f%%" % (percent_delta))

        # Testing weight deltas
        delta_w = back_conv_relu.weights.mem - conv_weights.reshape(2, 75)

        percent_delta_w = 100. * numpy.sum(numpy.abs(
            delta_w + conv_weight_delta.reshape(2, 75))) \
            / numpy.sum(numpy.abs(delta_w))

        # Hint: in CAFFE their \Delta W = - \Delta W_{Veles} (??)
        self.info("DELTA W: diff with CAFFE %.3f%%" % percent_delta_w)
        self.assertLess(percent_delta_w, max_percent_delta,
                        "Delta W differs by %.2f%%" % (percent_delta_w))

    def test_softmax(self, data_filename="softmax.txt"):
        """
        A complex test for EvaluatorSoftmax, All2AllSoftMax and GDSM layers.

        Args:
            data_filename(str): name of file with pooling data
                (relative from data dir)
        """
        n_classes = 10  # CIFAR
        n_pics = 2
        n_chans = 64
        size = 4

        max_percent_delta = 2.0

        lines = self._read_lines(data_filename)

        # Fwd prop
        a2a_bottom = self._read_array("a2a_bottom", lines,
                                      (n_pics, size, size, n_chans))
        a2a_top = self._read_array("a2a_top", lines, (n_pics, 1, 1, n_classes))
        a2a_weights_raw = self._read_array(
            "a2a_weights", lines, (n_classes, 1, size * size * n_chans, 1))
        a2a_weights_raw = a2a_weights_raw.reshape(
            n_classes, n_chans, size, size).swapaxes(1, 2).swapaxes(2, 3)

        a2a_weights = a2a_weights_raw.reshape(n_classes, size * size * n_chans)

        a2a_bias_raw = self._read_array("a2a_bias", lines,
                                        (1, 1, n_classes, 1))

        sm_bottom = self._read_array("sm_bottom", lines,
                                     (n_pics, 1, 1, n_classes))
        sm_top = self._read_array("sm_top", lines, (n_pics, 1, 1, n_classes))

        labels = self._read_array("labels", lines,
                                  (n_pics, 1, 1, 1)).astype(numpy.int32)

        a2a_softmax = all2all.All2AllSoftmax(
            self.parent, output_sample_shape=n_classes,
            weights_filling="uniform", weights_stddev=0.1,
            bias_filling="uniform", bias_stddev=0.01)

        a2a_softmax.input = Vector(a2a_bottom)

        a2a_softmax.initialize(self.device)
        a2a_softmax.weights.mem[:] = a2a_weights[:]

        a2a_softmax.weights.map_invalidate()
        a2a_softmax.bias.mem[:] = 0
        a2a_softmax.bias.map_invalidate()

        a2a_softmax.cpu_run()

        a2a_softmax.output.map_read()
        fwd_percent_delta = (numpy.sum(numpy.abs(
            a2a_softmax.output.mem - sm_top)) /
            (numpy.sum(numpy.abs(sm_top)))) * 100.

        self.info("A2A_SM FWD DELTA: %.3f%%" % fwd_percent_delta)
        self.assertLess(fwd_percent_delta, max_percent_delta,
                        "A2A_SM_FWD differs by %.2f%%" % (fwd_percent_delta))

        # Back prop

        sm_top_err = self._read_array("sm_top_diff", lines,
                                      (n_pics, 1, 1, n_classes))
        sm_bot_err = self._read_array("sm_bottom_diff", lines,
                                      (n_pics, 1, 1, n_classes))
        a2a_bot_err = self._read_array("a2a_bottom_diff", lines,
                                       (n_pics, size, size, n_chans))

        ev_sm = evaluator.EvaluatorSoftmax(self.parent)

        ev_sm.max_idx = a2a_softmax.max_idx
        ev_sm.batch_size = n_pics
        ev_sm.output = a2a_softmax.output
        ev_sm.labels = Vector(labels.reshape(n_pics))

        ev_sm.initialize(self.device)
        ev_sm.cpu_run()

        ev_sm.output.map_read()
        ev_sm.err_output.map_read()

        back_a2a_sm = gd.GDSoftmax(self.parent, store_gradient=False)

        back_a2a_sm.output = a2a_softmax.output
        back_a2a_sm.input = a2a_softmax.input
        back_a2a_sm.err_output = ev_sm.err_output
        back_a2a_sm.weights = a2a_softmax.weights
        back_a2a_sm.bias = a2a_softmax.bias

        back_a2a_sm.initialize(self.device)

        back_a2a_sm.cpu_run()

        back_a2a_sm.err_input.map_read()

        back_percent_delta = \
            100. * (numpy.sum(numpy.abs(
                back_a2a_sm.err_input.mem - a2a_bot_err)) /
                numpy.sum(numpy.abs(a2a_bot_err)))

        self.info("A2ASM_BACK_DELTA %.2f", back_percent_delta)

        manual_sm_bot_err = numpy.zeros(shape=(n_pics, 1, 1, n_classes),
                                        dtype=numpy.float64)
        for pic in range(n_pics):
            for i in range(n_classes):
                for j in range(n_classes):
                    if labels[pic] == j:
                        target = 1
                    else:
                        target = 0
                    manual_sm_bot_err[pic, 0, 0, i] += (
                        target / sm_top_err[pic, 0, 0, j] *
                        (sm_top_err[pic, 0, 0, i] * sm_top_err[pic, 0, 0, j]
                         - sm_top_err[pic, 0, 0, i] * int(i == j))
                        )

        manual_sm_bot_err /= n_pics  # WTF???!!!!
        self.info(" manual SM_BOT_ERR_DELTA %.3f%%" %
                  (numpy.sum(numpy.abs(manual_sm_bot_err - sm_bot_err))
                      / numpy.sum(numpy.abs(sm_bot_err),)))

        manual_a2a_bot_err = numpy.zeros(shape=(n_pics, size, size, n_chans),
                                         dtype=numpy.float64)
        for pic in range(n_pics):
            for cur_class in range(n_classes):
                for i in range(size):
                    for j in range(size):
                        for chan in range(n_chans):
                            manual_a2a_bot_err[pic, i, j, chan] += (
                                sm_bot_err[pic, 0, 0, cur_class] *
                                a2a_weights_raw[cur_class, i, j, chan])

        self.info(" manual A2A_BOT_ERR_DELTA %.3f%%" % (
            numpy.sum(numpy.abs(manual_a2a_bot_err - a2a_bot_err))
            / numpy.sum(numpy.abs(a2a_bot_err))))

        self.assertLess(back_percent_delta, max_percent_delta,
                        "Back A2SM differs by %.3f%%" % back_percent_delta)


if __name__ == "__main__":
    StandardTest.main()
