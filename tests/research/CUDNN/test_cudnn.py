
import cv2
import libcudnn as cudnn

import logging
import numpy

import pycuda.driver as cuda

import unittest
from zope.interface import implementer

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
from veles.dummy import DummyWorkflow

from veles import units
from veles.znicz import channel_splitting
from veles.znicz import conv


@implementer(units.IUnit)
class CUDNNConv(units.Unit):
    def __init__(self, workflow, **kwargs):
        super(CUDNNConv, self).__init__(workflow, **kwargs)
        self.input = None
        self.weights = None
        self.output = None

        self._n_pics = None
        self._n_chans = None
        self._pic_w = None
        self._pic_h = None

        self._n_filters = None
        self._n_filter_chans = None
        self._filter_h = None
        self._filter_w = None

        self._padding_h, self._padding_w = kwargs["padding"]
        self._stride_y, self._stride_x = kwargs["stride"]

        self._tensor_format = cudnn.cudnnTensorFormat["CUDNN_TENSOR_NCHW"]
        self._data_type = cudnn.cudnnDataType["CUDNN_DATA_DOUBLE"]
        self._conv_mode = cudnn.cudnnConvolutionMode["CUDNN_CROSS_CORRELATION"]
        self._accumulate = \
            cudnn.cudnnAccumulateResults["CUDNN_RESULT_NO_ACCUMULATE"]

        self._input_desc = None

        self._filters_desc = None

        self._dest_desc = None

        self._conv_desc = None
        self._conv_path = \
            cudnn.cudnnConvolutionPath["CUDNN_CONVOLUTION_FORWARD"]

        self._src_dev = None
        self._filter_dev = None

        self._cur_context = None

        self.output = formats.Vector()

    def initialize(self, device, **kwargs):
        self._n_pics, self._n_chans, self._pic_h, self._pic_w = \
            self.input.mem.shape

        self._n_filters, self._n_filter_chans, self._filter_h, \
            self._filter_w = self.weights.mem.shape

        self._handle = cudnn.cudnnCreate()

        self._input_desc = cudnn.cudnnCreateTensor4dDescriptor()

        cudnn.cudnnSetTensor4dDescriptor(
            self._input_desc, self._tensor_format, self._data_type,
            self._n_pics, self._n_chans, self._pic_h, self._pic_w)

        self._filters_desc = cudnn.cudnnCreateFilterDescriptor()
        cudnn.cudnnSetFilterDescriptor(
            self._filters_desc, self._data_type, self._n_filters,
            self._n_filter_chans, self._filter_h, self._filter_w)

        self._conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetConvolutionDescriptor(
            self._conv_desc, self._input_desc, self._filters_desc,
            self._padding_h, self._padding_w, self._stride_y, self._stride_x,
            1, 1, self._conv_mode)

        _, _, out_h, out_w = \
            cudnn.cudnnGetOutputTensor4dDim(self._conv_desc, self._conv_path)

        self._dest_desc = cudnn.cudnnCreateTensor4dDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(
            self._dest_desc, self._tensor_format, self._data_type,
            self._n_pics, self._n_filters, out_h, out_w)

        cuda.init()
        cur_device = cuda.Device(0)
        self._cur_context = cur_device.make_context()

        self.output.mem = numpy.zeros(
            shape=(self._n_pics, self._n_filters, out_h, out_w),
            dtype=self.input.mem.dtype)

    def __del__(self):
        self._cur_context.pop()

    def run(self):
        self.input.map_read()
        self._src_dev = cuda.to_device(self.input.mem)

        self.weights.map_read()
        self._filters_dev = cuda.to_device(self.weights.mem)

        self._dest_dev = cuda.mem_alloc(self.output.mem.nbytes)

        cudnn.cudnnConvolutionForward(
            self._handle, self._input_desc, int(self._src_dev),
            self._filters_desc, int(self._filters_dev), self._conv_desc,
            self._dest_desc, int(self._dest_dev),
            self._accumulate)

        dest_pic = cuda.from_device_like(self._dest_dev, self.output.mem)
        self.output.mem[:] = dest_pic[:]


class TestCUDNNBase(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        pass

    def get_cudnn_conv_result(self):
        workflow = DummyWorkflow()

        filters = numpy.zeros(shape=(1, 3, 3, 3), dtype=numpy.float64)
        for chan in range(filters.shape[3]):
            filters[0, :, :, chan] = numpy.array([[1, 1, 1], [1, 1, 1],
                                                 [1, 1, 1]])

        in_pic = cv2.imread("../../data/FM.jpg")

        workflow.end_point.unlink_all()

        input_data = numpy.array(in_pic, dtype=numpy.float64)
        input_data = input_data.reshape((1,) + input_data.shape)

        chan_splitter = channel_splitting.ChannelSplitter(workflow)
        chan_splitter.link_from(workflow.start_point)
        chan_splitter.input = formats.Vector(input_data)

        cudnn_conv = CUDNNConv(workflow, padding=(0, 0), stride=(1, 1))
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
        in_pic = cv2.imread("../data/FM.jpg")

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
