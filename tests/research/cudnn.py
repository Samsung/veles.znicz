import gc
import logging
import numpy
from PIL import Image
import unittest

import veles.memory as formats
import veles.backends as opencl
from veles.dummy import DummyWorkflow
from veles.znicz import cudnn

from veles.znicz import channel_splitting
from veles.znicz import conv


class TestCUDNNBase(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        gc.collect()
        del self.device

    def get_cudnn_conv_result(self):
        workflow = DummyWorkflow()

        filters = numpy.zeros(shape=(1, 3, 3, 3), dtype=numpy.float64)
        for chan in range(filters.shape[3]):
            filters[0, :, :, chan] = numpy.array([[1, 1, 1], [1, 1, 1],
                                                 [1, 1, 1]])

        in_pic = Image.open("../../data/FM.jpg")

        workflow.end_point.unlink_all()

        input_data = numpy.array(in_pic, dtype=numpy.float64)
        input_data = input_data.reshape((1,) + input_data.shape)

        chan_splitter = channel_splitting.ChannelSplitter(workflow)
        chan_splitter.link_from(workflow.start_point)
        chan_splitter.input = formats.Vector(input_data)

        cudnn_conv = cudnn.CUDNNConv(workflow, padding=(0, 0), stride=(1, 1))
        cudnn_conv.link_from(chan_splitter)
        cudnn_conv.link_attrs(chan_splitter, ("input", "output"))
        cudnn_conv.weights = formats.Vector(filters)

        chan_merger = channel_splitting.ChannelMerger(workflow)
        chan_merger.link_from(cudnn_conv)
        chan_merger.link_attrs(cudnn_conv, ("input", "output"))

        workflow.end_point.link_from(chan_merger)

        workflow.initialize(device=self.device)
        workflow.run()

        chan_merger.output.map_read()
        return chan_merger.output.mem.copy()

    def get_standard_conv_result(self):
        workflow = DummyWorkflow()
        filters = numpy.zeros(shape=(1, 3, 3, 3), dtype=numpy.float64)
        for chan in range(filters.shape[3]):
            filters[0, :, :, chan] = numpy.array([[1, 1, 1], [1, 1, 1],
                                                 [1, 1, 1]])
        in_pic = Image.open("../data/FM.jpg")

        workflow.end_point.unlink_all()

        input_data = numpy.array(in_pic, dtype=numpy.float64)
        input_data = input_data.reshape((1,) + input_data.shape)

        standard_conv = conv.Conv(workflow, n_kernels=1, kx=3, ky=3,
                                  sliding=(1, 1), padding=(0, 0, 0, 0))
        standard_conv.link_from(workflow.start_point)
        standard_conv.input = formats.Vector(input_data.copy())
        standard_conv.weights = formats.Vector(filters.reshape((1, 27)).copy())

        workflow.end_point.link_from(standard_conv)

        workflow.initialize(device=self.device)

        standard_conv.input.mem[:] = input_data[:]
        standard_conv.weights.mem[:] = filters.reshape((1, 27))[:]

        workflow.run()

        standard_conv.output.map_read()
        return standard_conv.output.mem.copy()

    def test_conv(self):
        standard_result = self.get_standard_conv_result()
        cudnn_result = self.get_cudnn_conv_result()
        delta = numpy.average((numpy.abs((standard_result - cudnn_result) /
                                         numpy.maximum(standard_result,
                                                       cudnn_result))))
        logging.info("CuDNN to Veles conv: %.3f%%" % (float(delta) * 100))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
