
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

        self.output = formats.Vector(numpy.zeros(
            shape=(self._n_pics, self._n_filters, out_h, out_w),
            dtype=self.input.mem.dtype))

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
        self.workflow = DummyWorkflow()

    def tearDown(self):
        pass

    def test_all(self):
        filters = numpy.zeros(shape=(1, 3, 3, 3), dtype=numpy.float64)
        for chan in range(filters.shape[3]):
            filters[0, :, :, chan] = numpy.array([[1, 1, 1], [1, 1, 1],
                                                 [1, 1, 1]])

        in_pic = cv2.imread("../data/FM.jpg")

        input_data = numpy.array(cv2.split(in_pic), dtype=numpy.float64)
        input_data = input_data.reshape((1,) + input_data.shape)

        cudnn_conv = CUDNNConv(self.workflow, padding=(0, 0), stride=(1, 1))
        cudnn_conv.link_from(self.workflow.start_point)
        self.workflow.end_point.unlink_all()
        self.workflow.end_point.link_from(cudnn_conv)

        cudnn_conv.input = formats.Vector(input_data)
        cudnn_conv.weights = formats.Vector(filters)

        self.workflow.initialize(device=self.device)
        self.workflow.run()

        cv2.imshow("FM2", cudnn_conv.output.mem[0][0] / 255. / 3 / 10)
        cv2.waitKey(0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
