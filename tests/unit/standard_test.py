"""
Created on June 06, 2014

A base class for test cases.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

# pylint: disable=W0633

import gc
import numpy as np
import os
import unittest

from veles import backends
from veles.dummy import DummyWorkflow


class StandardTest(unittest.TestCase):
    def setUp(self):
        self.workflow = DummyWorkflow()
        self.device = backends.Device()
        self.device.thread_pool_attach(self.workflow.thread_pool)
        self.data_dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data")

    def tearDown(self):
        del self.workflow
        gc.collect()
        del self.device

    def _read_array(self, array_name, lines, shape=None):
        """
        Reads a pic array from from export file, splitted to lines.
        NB: last line should be empty

        Args:
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

                out_array = np.zeros(shape=(n_pics, height, width, n_chans),
                                     dtype=np.float64)
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

        Args:
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
