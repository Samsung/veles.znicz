"""
Created on May 19, 2014

Unit test for convolutional layer forward propagation, compared to CAFFE data.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import logging
import numpy as np
import os
from scipy.signal import correlate2d, convolve2d  # pylint: disable=E0611
import unittest


from veles import opencl
from veles.formats import Vector
import veles.znicz.conv as conv
import veles.znicz.pooling as pooling
import veles.znicz.gd_conv as gd_conv
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.normalization as normalization
from veles.tests.dummy_workflow import DummyWorkflow

os.environ["PYOPENCL_CTX"] = "1:0"  # Uncomment to change OpenCL device


class TestConvCaffe(unittest.TestCase):
    def setUp(self):
        self.workflow = DummyWorkflow()
        self.device = opencl.Device()
        self.data_dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data")

    def tearDown(self):
        pass

    def _read_array(self, array_name, lines, shape):
        """
        Reads a pic array from from export file, splitted to lines.
        NB: last line should be empty

        Args:
            array_name(str): name of array to read
        lines(array): lines of file to read from
        shape(tuple): array shape=(n_pics, height, width, n_chans)

        Returns:
            :class:`numpy.ndarray`

        """
        n_pics, height, width, n_chans = shape

        out_array = np.zeros(shape=shape, dtype=np.float64)

        cur_line = None
        for i, line in enumerate(lines):
            line = line.replace("\n", "")
#            print([array_name, line])
            if line == array_name:

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
#                    print(lines[cur_line].split("\t"))
                    data = [float(x) for x in lines[cur_line].split("\t")]
                    cur_line += 1

                    for j in range(width):
                        out_array[cur_pic, i, j, cur_chan] = data[j]
        return out_array

    def test_caffe_conv(self, data_filename="conv.txt"):
        """
        Compare CAFFE conv layer fwd prop with Veles conv layer.

        Args:
            data_filename(str): name to file with pooling data,
                exported from CAFFE (searched in ``self.data_dir_path``)
        """
        in_file = open(os.path.join(self.data_dir_path, data_filename), 'r')
        lines = in_file.readlines()
        in_file.close()

        kernel_size = 5
        padding_size = 2

        bottom = self._read_array("bottom", lines=lines, shape=(2, 32, 32, 3))
        weights = self._read_array("weights", lines=lines, shape=(2, 5, 5, 3))
        top = self._read_array("top", lines=lines, shape=(2, 32, 32, 2))

        fwd_conv = conv.Conv(self.workflow, kx=kernel_size, ky=kernel_size,
                             padding=(padding_size, padding_size,
                                      padding_size, padding_size),
                             sliding=(1, 1),
                             n_kernels=2)

        fwd_conv.input = Vector()
        fwd_conv.input.mem = bottom

        # UNCOMMENT TO SEE CAFFEE DATA
#        print("bottom shape:", bottom.shape)
#        print(bottom)
#        print("weights shape:", weights.shape)
#        print(weights)
#        print("top shape:", top.shape)
#        print(top)

        fwd_conv.initialize(self.device)
        fwd_conv.weights.map_invalidate()
        fwd_conv.weights.mem[:] = weights.reshape(2, 75)[:]
        fwd_conv.bias.map_invalidate()
        fwd_conv.bias.mem[:] = 0
        fwd_conv.run()

        logging.info("Veles vs CAFFE data:")
        fwd_conv.output.map_read()

        logging.info("Veles top shape:" + str(fwd_conv.output.mem.shape))
        delta_with_veles = fwd_conv.output.mem - top

        logging.info("CONV: diff with Veles: %.2f%%" % (100. * np.sum(np.abs(
            delta_with_veles)) / np.sum(np.abs(fwd_conv.output.mem)),))

        logging.info("COMPARED TO HANDMADE CORRELATION:")
        scipy_conv_out = np.zeros(shape=(2, 32, 32, 2), dtype=np.float64)

        for pic in range(2):
            for color_chan in range(3):
                for weight_id in range(2):
                    correlation = correlate2d(
                        bottom[pic, :, :, color_chan],
                        weights[weight_id, :, :, color_chan], mode="same")
                    scipy_conv_out[pic, :, :, weight_id] += correlation

        delta_with_scipy = fwd_conv.output.mem - scipy_conv_out
        logging.info("CONV: diff with SciPy: %.2f%%" % (100. * np.sum(np.abs(
            delta_with_scipy)) / np.sum(np.abs(fwd_conv.output.mem)),))

    def test_caffe_grad_conv(self, data_filename="conv_grad.txt"):
        """
        Compare CAFFE conv layer with Veles conv layer (FwdProp and BackProp).

        Args:
            data_filename(str): name to file with pooling data,
                exported from CAFFE (searched in ``self.data_dir_path``)
        """
        in_file = open(os.path.join(self.data_dir_path, data_filename), 'r')
        lines = in_file.readlines()
        in_file.close()

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

        fwd_conv = conv.Conv(self.workflow, kx=kernel_size, ky=kernel_size,
                             padding=(padding_size, padding_size,
                                      padding_size, padding_size),
                             sliding=(1, 1), n_kernels=n_kernels)

        fwd_conv.input = Vector(bottom)
#
#        #UNCOMMENT TO SEE CAFFEE DATA
# #        print("bottom shape:", bottom.shape)
# #        print(bottom)
# #        print("weights shape:", weights.shape)
# #        print(weights)
# #        print("top shape:", top.shape)
# #        print(top)

        fwd_conv.initialize(self.device)
        fwd_conv.weights.map_invalidate()
        fwd_conv.weights.mem[:] = weights.reshape(2, 75)[:]
        fwd_conv.bias.map_invalidate()
        fwd_conv.bias.mem[:] = 0
        fwd_conv.run()

        logging.info("Veles vs CAFFE data:")
        fwd_conv.output.map_read()

        logging.info("Veles top shape:" + str(fwd_conv.output.mem.shape))
        delta_with_veles = fwd_conv.output.mem - top

        logging.info("CONV: diff with CAFFE: %.2f%%" % (100. * np.sum(np.abs(
            delta_with_veles)) / np.sum(np.abs(fwd_conv.output.mem)),))

        back_conv = gd_conv.GradientDescentConv(self.workflow, kx=kernel_size,
                                                padding=(padding_size,
                                                         padding_size,
                                                         padding_size,
                                                         padding_size),
                                                ky=kernel_size, sliding=(1, 1),
                                                n_kernels=n_kernels,
                                                device=self.device)

        back_conv.input = Vector(bottom)

        back_conv.output = Vector(top)

        back_conv.err_output = Vector(top_err)

        back_conv.weights = Vector()
        back_conv.weights.mem = fwd_conv.weights.mem

        back_conv.bias = Vector(fwd_conv.bias.mem)

        back_conv.batch_size = 2

        back_conv.initialize(device=self.device)

        back_conv.gpu_err_input_update()

        back_conv.err_input.map_read()

        # BACKPROP: difference with CAFFE export
        back_delta = back_conv.err_input.mem - bot_err
#        print(bot_err)
#        print("~~~~~~~~~~")
#        print(back_conv.err_input.mem)

        logging.info("GDCONV: diff with CAFFE: %.3f%%" %
                     (100. * np.sum(np.fabs(back_delta)) /
                      np.sum(np.fabs(back_conv.err_input.mem)),))

        # perform manual GD CONV
        manual_bot_err = np.zeros(shape=(2, bot_size, bot_size, 3),
                                  dtype=np.float64)
        for pic in range(batch_size):
            for color_chan in range(3):
                for weight_id in range(n_kernels):
                    conv_result = convolve2d(
                        top_err[pic, :, :, weight_id],
                        weights[weight_id, :, :, color_chan], mode="same")
                    manual_bot_err[pic, :, :, color_chan] += conv_result

        caffe_to_manual_delta = manual_bot_err - bot_err
        logging.info("Manual GDCONV: diff with CAFFE: %.3f%%" % (
            100. * np.sum(np.fabs(manual_bot_err - bot_err)) /
            np.sum(np.fabs(bot_err))))

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
        top = self._read_array("top", lines=lines,
                               shape=(2, out_height, out_width, 2))

        # do pooling with VELES
        fwd_pool = pooling.MaxPooling(self.workflow, kx=kernel_size,
                                      ky=kernel_size, sliding=(stride, stride),
                                      device=self.device)
        fwd_pool.input = Vector(bottom)
        fwd_pool.input.map_write()

        fwd_pool.initialize(device=self.device)

        fwd_pool.cpu_run()
        fwd_pool.output.map_read()

        # UNCOMMENT TO SEE EXTRA DEBUG DATA
#        print(fwd_pool.output.mem)
#        logging.info("top")
#        print(top)

#        print(np.sum(np.abs(fwd_pool.output.mem - top)) / np.sum(np.abs(top)))
#        print(fwd_pool.output.mem - top)

        # do MANUAL pooling
        manual_pooling_out = np.zeros(shape=(2, out_height, out_width, 2),
                                      dtype=np.float64)
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
                                           chan] = np.max((zone))

    def test_caffe_grad_pooling(self, data_filename="pool_grad.txt"):
        """
        Compare CAFFE pooling unit with Veles ones (fwd and back propagations)

        Args:
            data_filename(str): name to file with pooling data,
                exported from CAFFE (searched in ``self.data_dir_path``)
        """
        bot_size = 32
        top_size = 16
        kernel_size = 3
        stride = 2
        n_chans = 2
        n_pics = 2

        in_file = open(os.path.join(self.data_dir_path, data_filename), 'r')
        lines = in_file.readlines()
        in_file.close()

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
        fwd_pool = pooling.MaxPooling(self.workflow, kx=kernel_size,
                                      ky=kernel_size, sliding=(stride, stride))
        fwd_pool.input = Vector(bottom)
        fwd_pool.input.map_write()

        fwd_pool.initialize(device=self.device)

        fwd_pool.cpu_run()
        fwd_pool.output.map_read()

        logging.info("FWD POOL: Veles vs CAFFE: %.3f%%" %
                     (100. * (np.sum(np.abs(fwd_pool.output.mem - top)) /
                              np.sum(np.abs(top)))))

        # Do MANUAL pooling
        out_height, out_width = top_size, top_size
        manual_pooling_out = np.zeros(shape=(2, out_height, out_width, 2),
                                      dtype=np.float64)
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
                                           chan] = np.max((zone))

        # BACK PROP
        grad_pool = gd_pooling.GDMaxPooling(self.workflow, kx=kernel_size,
                                            ky=kernel_size,
                                            sliding=(stride, stride),
                                            device=self.device)
        grad_pool.input = Vector(bottom)
        grad_pool.input.map_write()

        grad_pool.err_output = Vector(top_err)
        grad_pool.err_output.map_write()

        grad_pool.input_offs = fwd_pool.input_offs

        grad_pool.initialize(device=self.device)

        grad_pool.cpu_run()

        grad_pool.err_input.map_read()
        logging.info("BACK POOL: Veles vs CAFFE, %.3f%%" % (100 * np.sum(
            np.abs(grad_pool.err_input.mem - bot_err)) /
            np.sum(np.abs(bot_err))))

    def test_caffe_grad_normalization(self, data_filename="norm_gd.txt"):
        """
        Tests LRU normalization unit: compares it with CAFFE one.
        Fwd and back props made.

        Args:
            data_filename(str): name to file with pooling data,
                exported from CAFFE (searched in ``self.data_dir_path``)
        """
        in_file = open(os.path.join(self.data_dir_path, data_filename), 'r')
        lines = in_file.readlines()
        in_file.close()

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

        fwd_norm = normalization.LRNormalizerForward(self.workflow,
                                                     k=1, device=self.device)

        # FWD PROP
        fwd_norm.input = Vector(bottom)

        fwd_norm.initialize(self.device)
        fwd_norm.ocl_run()

        fwd_norm.output.map_read()

        fwd_percent_delta = 100. * (np.sum(np.abs(fwd_norm.output.mem - top)) /
                                    np.sum(np.abs(top)))

        logging.info("FWD NORM DELTA: %.2f%%" % fwd_percent_delta)
        self.assertLess(fwd_percent_delta, max_percent_delta,
                        "Fwd norm differs by %.2f%%" % (fwd_percent_delta))

        # BACK PROP
        back_norm = normalization.LRNormalizerBackward(self.workflow,
                                                       k=1, device=self.device)
        back_norm.output = Vector(top)

        back_norm.input = Vector(bottom)

        back_norm.err_output = Vector(top_err)

        back_norm.initialize(self.device)
        back_norm.ocl_run()

        back_norm.err_input.map_read()

        back_percent_delta = 100. * np.sum(
            np.abs(back_norm.err_output.mem - bot_err)) / \
            np.sum(np.abs(bot_err))

        logging.info("BACK NORM DELTA: %.2f%%" % back_percent_delta)
        self.assertLess(back_percent_delta, max_percent_delta,
                        "Fwd norm differs by %.2f%%" % (back_percent_delta))

    def test_caffe_relu(self, data_filename="conv_relu.txt"):
        """
        Tests CONV+RELU unit: compares it with CAFFE one.
        Fwd prop only.

        Args:
            data_filename(str): name to file with pooling data,
                exported from CAFFE (searched in ``self.data_dir_path``)
        """
        in_file = open(os.path.join(self.data_dir_path, data_filename), 'r')
        lines = in_file.readlines()
        in_file.close()

        in_size = 32
        out_size = 32
        n_chans = 3
        n_kernels = 2
        n_pics = 2
        kernel_size = 5
        padding = 2

        max_percent_delta = 2.0

        conv_bottom = self._read_array(
            "conv_bottom", lines, shape=(n_pics, in_size, in_size, n_chans))
        conv_top = self._read_array(
            "conv_top", lines, shape=(n_pics, out_size, out_size, n_kernels))
        relu_top_flat = self._read_array(
            "relu_top_flat", lines, shape=(1, 1, conv_top.size, 1)).ravel()
        relu_top = np.ndarray(
            shape=(n_pics, out_size, out_size, n_kernels), dtype=np.float64)
        cur_pos = 0
        for pic in range(n_pics):
            for kernel in range(n_kernels):
                for i in range(out_size):
                    for j in range(out_size):
                        relu_top[pic, i, j, kernel] = relu_top_flat[cur_pos]
                        cur_pos += 1

        conv_weights = self._read_array(
            "conv_weights", lines, shape=(n_kernels, kernel_size, kernel_size,
                                          n_chans))

        fwd_conv_relu = conv.ConvStrictRELU(self.workflow,
                                            kx=kernel_size,
                                            ky=kernel_size,
                                            padding=(padding, padding,
                                                     padding, padding),
                                            sliding=(1, 1),
                                            n_kernels=n_kernels,
                                            device=self.device)

        fwd_conv_relu.input = Vector(conv_bottom)

        fwd_conv_relu.initialize(self.device)

        fwd_conv_relu.weights.map_invalidate()
        fwd_conv_relu.weights.mem[:] = conv_weights.reshape(2, 75)[:]
        fwd_conv_relu.bias.map_invalidate()
        fwd_conv_relu.bias.mem[:] = 0

        fwd_conv_relu.ocl_run()

        fwd_conv_relu.output.map_read()

        percent_delta = 100. * (np.sum(np.abs(
            fwd_conv_relu.output.mem - relu_top)) / np.sum(np.abs(relu_top)))

        logging.info("CONV_RELU: diff with CAFFE %.3f%%" % percent_delta)
        self.assertLess(percent_delta, max_percent_delta,
                        "Fwd ConvRELU differs by %.2f%%" % (percent_delta))

        relu_top_manual = np.where(np.greater(conv_top, 0), conv_top, 0)
        manual_delta = 100. * np.sum(
            np.abs(relu_top_manual - relu_top)) / (np.sum(np.abs(relu_top)))
        logging.info("SciPy CONV_RELU: diff with CAFFE %.3f%%" % manual_delta)

    def test_caffe_grad_relu(self, data_filename="conv_relu.txt"):
        """
        Tests CONV+RELU unit: compares it with CAFFE one.
        Fwd prop only.

        Args:
            data_filename(str): name to file with pooling data,
                exported from CAFFE (searched in ``self.data_dir_path``)
        """
        in_file = open(os.path.join(self.data_dir_path, data_filename), 'r')
        lines = in_file.readlines()
        in_file.close()

        in_size = 32
        out_size = 32
        n_chans = 3
        n_kernels = 2
        n_pics = 2
        kernel_size = 5
        padding_size = 2

        max_percent_delta = 2.0

        conv_bottom = self._read_array("conv_bottom", lines,
                                  shape=(n_pics, in_size, in_size, n_chans))
        conv_top = self._read_array("conv_top", lines, shape=(n_pics,
                                                out_size, out_size, n_kernels))
        relu_top_flat = self._read_array("relu_top_flat", lines,
                                        shape=(1, 1, conv_top.size, 1)).ravel()
        relu_top = np.ndarray(shape=(n_pics, out_size, out_size, n_kernels),
                                                            dtype=np.float64)
        cur_pos = 0
        for pic in range(n_pics):
            for kernel in range(n_kernels):
                for i in range(out_size):
                    for j in range(out_size):
                        relu_top[pic, i, j, kernel] = relu_top_flat[cur_pos]
                        cur_pos += 1

        conv_weights = self._read_array("conv_weights", lines, shape=(
                                n_kernels, kernel_size, kernel_size, n_chans))

        fwd_conv_relu = conv.ConvStrictRELU(self.workflow,
                                  kx=kernel_size, ky=kernel_size,
                                  padding=(padding_size, padding_size,
                                           padding_size, padding_size),
                                  sliding=(1, 1), n_kernels=n_kernels,
                                  device=self.device)

        fwd_conv_relu.input = Vector(conv_bottom)

        fwd_conv_relu.initialize(self.device)

        fwd_conv_relu.weights.map_invalidate()
        fwd_conv_relu.weights.mem[:] = conv_weights.reshape(2, 75)[:]
        fwd_conv_relu.bias.map_invalidate()
        fwd_conv_relu.bias.mem[:] = 0

        fwd_conv_relu.ocl_run()

        fwd_conv_relu.output.map_read()

        percent_delta = 100. * (np.sum(np.abs(
            fwd_conv_relu.output.mem - relu_top)) / np.sum(np.abs(relu_top)))

        logging.info("CONV_RELU: diff with CAFFE %.3f%%" % percent_delta)
        self.assertLess(percent_delta, max_percent_delta,
                        "Fwd ConvRELU differs by %.2f%%" % (percent_delta))

        relu_top_manual = np.where(np.greater(conv_top, 0), conv_top, 0)
        manual_delta = 100. * np.sum(np.abs(relu_top_manual - relu_top)) \
                                                / (np.sum(np.abs(relu_top)))
        logging.info("SciPy CONV_RELU: diff with CAFFE %.3f%%" % manual_delta)

    def test_grad_conv_relu(self, data_filename="conv_relu_grad.txt"):
        """
        Tests GDDRELU_CONV unit: compares it with CAFFE one.
        Backward prop only

        Args:
            data_filename(str): name to file with pooling data,
                exported from CAFFE (searched in ``self.data_dir_path``)
        """
        in_file = open(os.path.join(self.data_dir_path, data_filename), 'r')
        lines = in_file.readlines()
        in_file.close()

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
        conv_bottom = self._read_array("conv_bottom", lines, shape=(n_pics,
                                                in_size, in_size, n_chans))

        conv_weights = self._read_array("conv_weights", lines,
                        shape=(n_kernels, kernel_size, kernel_size, n_chans))

        relu_top_err = self._read_array("relu_top_diff", lines, shape=(n_pics,
                                                in_size, in_size, n_kernels))

        relu_top_flat = self._read_array("relu_top_flat", lines,
                                    shape=(1, 1, relu_top_err.size, 1)).ravel()

        relu_top = np.ndarray(shape=(n_pics, out_size, out_size, n_kernels),
                                                            dtype=np.float64)
        cur_pos = 0
        for pic in range(n_pics):
            for kernel in range(n_kernels):
                for i in range(out_size):
                    for j in range(out_size):
                        relu_top[pic, i, j, kernel] = relu_top_flat[cur_pos]
                        cur_pos += 1

        back_conv_relu = gd_conv.GDStrictRELUConv(self.workflow,
                                                  kx=kernel_size,
                                                  ky=kernel_size,
                                                  padding=(padding, padding,
                                                           padding, padding),
                                                  sliding=(1, 1),
                                                  n_kernels=n_kernels,
                                                  device=self.device)

        back_conv_relu.batch_size = n_pics

        back_conv_relu.err_output = Vector(relu_top_err)
        back_conv_relu.input = Vector(conv_bottom)

        back_conv_relu.weights = Vector(conv_weights.reshape(2, 75))
        back_conv_relu.bias = Vector(np.zeros(shape=n_kernels))

        back_conv_relu.output = Vector(relu_top)

        back_conv_relu.initialize(device=self.device)

        back_conv_relu.weights.map_invalidate()
        back_conv_relu.weights.mem[:] = conv_weights.reshape(2, 75)[:]
        back_conv_relu.bias.map_invalidate()
        back_conv_relu.bias.mem[:] = 0

        back_conv_relu.cpu_run()

        back_conv_relu.err_input.map_read()
        result = back_conv_relu.err_input.mem

        percent_delta = 100. * (np.sum(np.abs(result - conv_bot_err)) \
                                / np.sum(np.abs(result)))
        logging.info("GD_CONV_RELU: diff with CAFFE %.3f%%" % percent_delta)
        self.assertLess(percent_delta, max_percent_delta,
                        "Fwd GD_ConvRELU differs by %.2f%%" % (percent_delta))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("CAFFE CONV TEST")
    unittest.main()
