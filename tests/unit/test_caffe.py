"""
Created on May 19, 2014

Unit test for convolutional layer forward propagation, compared to CAFFE data.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import logging
from veles import opencl
import unittest
from veles.formats import Vector

import veles.znicz.conv as conv
import veles.znicz.pooling as pooling
from veles.tests.dummy_workflow import DummyWorkflow

from scipy.signal import correlate2d

import numpy as np


class TestConvCaffe(unittest.TestCase):
    def setUp(self):
        self.workflow = DummyWorkflow()
        self.device = opencl.Device()

    def tearDown(self):
        pass

    def _read_array(self, array_name, lines, shape):
        """
        Reads a pic array from from export file, splitted to lines.

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
            if line == array_name:
                cur_line = i + 1

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

    def test_caffe_conv(self, data_path="data/convtest.txt"):
        """
        Compare CAFFE conv layer fwd prop with Veles conv layer.

        Args:
            data_path(str): path to file with data, exported from CAFFE
        """
        in_file = open(data_path, 'r')
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

        #UNCOMMENT TO SEE CAFFEE DATA
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

        logging.info("Difference with Veles: %.2f%%" % (100. *np.sum(np.abs(
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
        logging.info("Difference with SciPy: %.2f%%" % (100. * np.sum(np.abs(
            delta_with_scipy)) / np.sum(np.abs(fwd_conv.output.mem)),))

    def test_caffe_pooling(self, data_path="/home/agolovizin/pooltest.txt"):
        """
        Compare CAFFE pooling unit fwd_prop with Veles one

        Args:
            data_path(str): path to file with pooling data, exported from CAFFE
        """

        #load pooling data from CAFFE dumps
        in_file = open(data_path, 'r')
        lines = in_file.readlines()
        in_file.close()

        #max pooling: 3x3 kernel, 2x2 stride
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
                                      ky=kernel_size, sliding=(stride, stride))
        fwd_pool.input = Vector()
        fwd_pool.input.mem = bottom
        fwd_pool.input.map_write()

        fwd_pool.initialize(device=self.device)

        fwd_pool.run()
        fwd_pool.output.map_read()

        # UNCOMMENT TO SEE EXTRA DEBUG DATA
#        print(fwd_pool.output.mem)
#        logging.info("top")
#        print(top)

#        print(np.sum(np.abs(fwd_pool.output.mem - top)) / np.sum(np.abs(top)))
#        print(fwd_pool.output.mem - top)

        #do MANUAL pooling
        manual_pooling_out = np.zeros(shape=(2, out_height, out_width, 2),
                                      dtype=np.float64)
        for pic in range(2):
            for chan in range(2):
                for i_out in range(out_height):
                    for j_out in range(out_width):
                        min_i = i_out * 2
                        max_i = i_out * 2 + 3
                        min_j = j_out * 2
                        max_j = j_out * 2 + 3

                        zone = bottom[pic, min_i: max_i, min_j: max_j, chan]
                        manual_pooling_out[pic, i_out, j_out, chan] = np.max(
                                                                (zone))

#        logging.info("CAFFE_top - manual")
#        print(top - manual_pooling_out)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("CAFFE CONV TESTЪ")
    unittest.main()
