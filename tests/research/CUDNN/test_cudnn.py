
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

from veles import error
from veles import units

from veles.znicz import channel_splitting, gd_conv
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

        self.output.mem = numpy.zeros(
            shape=(self._n_pics, self._n_filters, out_h, out_w),
            dtype=self.input.mem.dtype)

    def __del__(self):
        self._cur_context.pop()
        pass

    def run(self):

        cuda.init()
        cur_device = cuda.Device(0)
        self._cur_context = cur_device.make_context()

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


@implementer(units.IUnit)
class CUDNNConvBackward(units.Unit):
    def __init__(self, workflow, **kwargs):
        super(CUDNNConvBackward, self).__init__(workflow, **kwargs)
        try:
            n_kernels = kwargs["n_kernels"]
            kx = kwargs["kx"]
            ky = kwargs["ky"]
        except KeyError:
            raise KeyError("n_kernels, kx and ky are required parameters")
        padding = kwargs.get("padding", (0, 0))  # Left Top Right Bottom
        sliding = kwargs.get("sliding", (1, 1))  # Y X
        kwargs["n_kernels"] = n_kernels
        kwargs["kx"] = kx
        kwargs["ky"] = ky
        kwargs["padding"] = padding
        kwargs["sliding"] = sliding

        self.n_kernels = n_kernels
        self.kx = kx
        self.ky = ky
        self.padding = tuple(padding)
        self.sliding = tuple(sliding)

        self._handle = None
        self._tensor_format = cudnn.cudnnTensorFormat["CUDNN_TENSOR_NCHW"]
        self._data_type = cudnn.cudnnDataType["CUDNN_DATA_DOUBLE"]
        self._conv_mode = cudnn.cudnnConvolutionMode["CUDNN_CROSS_CORRELATION"]
        self._accumulate = \
            cudnn.cudnnAccumulateResults["CUDNN_RESULT_NO_ACCUMULATE"]

        self._bias_desc = None
        self._bias_dev = None
        self._gradient_bias_desc = None
        self._gradient_bias_dev = None

        self._cur_context = None
        self._cur_device = None

        self.err_intput = formats.Vector()
        self.err_output = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.weights = None
        self.bias = None
        self.input = None
        self.output = None

        self._cuda_initialized = False

    def initialize(self, device, **kwargs):
        if self.err_output.shape != self.output.shape:
            raise error.BadFormatError("err_output.shape != output.shape")

        self._n_pics, self._n_chans, self._pic_h, self._pic_w = \
            self.input.mem.shape

        self._n_filters, self._n_filter_chans, self._filter_h, \
            self._filter_w = self.weights.mem.shape

        self.err_input = formats.Vector(numpy.zeros_like(self.input.mem))
        _, self._n_filters, self._out_pic_h, self._out_pic_w = \
            self.err_output.shape

        self._gradient_bias_desc = cudnn.cudnnCreateTensor4dDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(
            self._gradient_bias_desc, self._tensor_format, self._data_type,
            1, self.n_kernels, 1, 1)

        self._bias_desc = cudnn.cudnnCreateTensor4dDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(
            self._bias_desc, self._tensor_format, self._data_type,
            self.n_kernels, 1, 1, 1)

        self._err_output_desc = cudnn.cudnnCreateTensor4dDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(
            self._err_output_desc, self._tensor_format, self._data_type,
            self._n_pics, self._n_filters, self._out_pic_h, self._out_pic_w)

        self._input_desc = cudnn.cudnnCreateTensor4dDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(
            self._input_desc, self._tensor_format, self._data_type,
            *self.input.mem.shape)

        self._filters_desc = cudnn.cudnnCreateFilterDescriptor()
        cudnn.cudnnSetFilterDescriptor(
            self._filters_desc, self._data_type, self._n_filters,
            self._n_chans, self._filter_h, self._filter_w)

        self._conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetConvolutionDescriptor(
            self._conv_desc, self._input_desc, self._filters_desc,
            self.padding[0], self.padding[1],
            self.sliding[0], self.sliding[1],
            1, 1, self._conv_mode)

        n, c, h, w = cudnn.cudnnGetOutputTensor4dDim(
            self._conv_desc, cudnn.cudnnConvolutionPath[
                "CUDNN_CONVOLUTION_DATA_PATH"])
        assert (n, c, h, w) == self.input.shape

    def _init_cuda(self):
        cuda.init()
        self._cur_device = cuda.Device(0)
        self._cur_context = self._cur_device.make_context()
        self._handle = cudnn.cudnnCreate()
        self._cuda_initialized = True

    def run(self):
        if not self._cuda_initialized:
            self._init_cuda()

        self.bias.map_read()

        # bias
        self._bias_dev = cuda.to_device(self.bias.mem)
        self._gradient_bias_dev = cuda.to_device(self.gradient_bias.mem)

        self.err_output.map_read()
        err_output_dev = cuda.to_device(self.err_output.mem)

        cudnn.cudnnConvolutionBackwardBias(
            self._handle, self._err_output_desc, int(err_output_dev),
            self._gradient_bias_desc, int(self._gradient_bias_dev),
            self._accumulate)

        self.gradient_bias.map_invalidate()
        self.gradient_bias.mem[:] = cuda.from_device_like(
            self._gradient_bias_dev, self.gradient_bias.mem)[:]

        # data
        self.input.map_read()
        input_dev = cuda.to_device(self.input.mem)

        self.weights.map_read()
        filters_dev = cuda.to_device(self.weights.mem)

        err_input_dev = cuda.mem_alloc_like(self.input.mem)

        assert self.input.shape == self.err_input.shape

        cudnn.cudnnConvolutionBackwardData(
            self._handle, self._filters_desc, int(filters_dev),
            self._err_output_desc, int(err_output_dev),
            self._conv_desc,
            self._input_desc, int(err_input_dev),
            self._accumulate)

        self.err_input.map_invalidate()
        self.err_input.mem[:] = cuda.from_device(
            err_input_dev, self.err_input.mem.shape, self.input.mem.dtype)[:]

        # filters
        n, c, h, w = cudnn.cudnnGetOutputTensor4dDim(
            self._conv_desc,
            cudnn.cudnnConvolutionPath["CUDNN_CONVOLUTION_WEIGHT_GRAD"])

        self.gradient_weights.map_read()
        grad_weights_dev = cuda.mem_alloc_like(self.gradient_weights.mem)
        cudnn.cudnnConvolutionBackwardFilter(
            self._handle, self._input_desc, int(input_dev),
            self._err_output_desc, int(err_output_dev),
            self._conv_desc, self._filters_desc, int(grad_weights_dev),
            self._accumulate)

        assert n * c * h * w == self.gradient_weights.mem.size
        grad_weights = cuda.from_device(
            grad_weights_dev, shape=(n, c, h, w),
            dtype=self.gradient_weights.mem.dtype)

        self.gradient_weights.map_invalidate()
        self.gradient_weights.mem[:] = grad_weights.reshape(
            self.gradient_weights.mem.shape)[:]

    def ocl_run(self):
        logging.info("OCL RUN")

    def cpu_run(self):
        logging.info("CPU_RUN")

    def __del__(self):
        self._cur_context.pop()


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
        in_pic = cv2.imread("../../data/FM.jpg")

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

    def _relative_delta(self, array1, array2):
        a = numpy.average(array1 - array2)
        b = numpy.average(numpy.abs(numpy.maximum(numpy.abs(array1),
                                                  numpy.abs(array2))))
        return a / b

    def test_fwd_conv(self):
        standard_result = self.get_standard_conv_result()
        cudnn_result = self.get_cudnn_conv_result()
        delta = self._relative_delta(standard_result, cudnn_result)
        logging.info("CuDNN to Veles conv: %.3f%%" % (float(delta) * 100))

    def _load_array(self, array_name):
        return numpy.load("../../data/gd_conv_data/gd_conv3.%s.npz" %
                          array_name).items()[0][1]

    def _load_array_nchw(self, array_name):
        return self._load_array(array_name).swapaxes(3, 2).swapaxes(2, 1)

    def _cudnn_back_conv(self):
        input = self._load_array_nchw("input")

        output = self._load_array_nchw("output")

        err_output = self._load_array_nchw("err_output")

        kx, ky = 5, 5
        padding = (2, 2)
        sliding = (1, 1)
        n_kernels = 64

        weights = numpy.load(
            "../../data/gd_conv_data/gd_conv3.weights.npz").items()[0][1]
        n_chans = int(weights.shape[1] / 5 / 5)
        weights = weights.reshape((n_kernels, ky, kx, n_chans))
        weights = weights.swapaxes(3, 2).swapaxes(2, 1)

        gradient_weights = numpy.load(
            "../../data/gd_conv_data/gd_conv3.gradient_weights.npz").items(
            )[0][1].reshape((n_kernels, ky, kx, n_chans)).swapaxes(
            3, 2).swapaxes(2, 1)

        bias = numpy.load(
            "../../data/gd_conv_data/gd_conv3.bias.npz").items()[0][1]

        gradient_bias = numpy.load(
            "../../data/gd_conv_data/gd_conv3.gradient_bias.npz").items(
            )[0][1]

        workflow = DummyWorkflow()
        cudnn_gd_conv = CUDNNConvBackward(
            workflow, n_kernels=n_kernels, kx=kx, ky=ky, padding=padding,
            sliding=sliding)
        cudnn_gd_conv.link_from(workflow.start_point)

        cudnn_gd_conv.input = formats.Vector(input)
        cudnn_gd_conv.output = formats.Vector(output)
        cudnn_gd_conv.err_output = formats.Vector(err_output)
        cudnn_gd_conv.weights = formats.Vector(weights.copy())
        cudnn_gd_conv.bias = formats.Vector(bias.copy())
        cudnn_gd_conv.gradient_weights = formats.Vector(gradient_weights)
        cudnn_gd_conv.gradient_bias = formats.Vector(gradient_bias)

        workflow.end_point.link_from(cudnn_gd_conv)

        workflow.initialize(device=self.device)

        cudnn_gd_conv.weights.map_invalidate()
        cudnn_gd_conv.weights.mem[:] = weights[:]
        cudnn_gd_conv.bias.map_invalidate()
        cudnn_gd_conv.bias.mem[:] = bias[:]

        cudnn_gd_conv.run()

        gd_bias = cudnn_gd_conv.gradient_bias.mem.swapaxes(1, 2).swapaxes(2, 3)
        err_input = cudnn_gd_conv.err_input.mem.swapaxes(1, 2).swapaxes(2, 3)
        gd_weights = cudnn_gd_conv.gradient_weights.mem.swapaxes(
            1, 2).swapaxes(2, 3)

        return err_input, -gd_weights, -gd_bias

    def _veles_back_conv(self):
        workflow = DummyWorkflow()

        input = self._load_array("input")
        output = self._load_array("output")
        err_output = self._load_array("err_output")

        bias = self._load_array("bias")
        weights = self._load_array("weights")

        kx, ky = 5, 5
        padding = (2, 2, 2, 2)
        sliding = (1, 1)
        n_kernels = 64

        veles_gd_conv = gd_conv.GradientDescentConv(
            workflow, kx=kx, ky=ky, padding=padding, sliding=sliding,
            n_kernels=n_kernels,
            batch_size=1, learning_rate=1, learning_rate_bias=1,
            weights_decay=0, weights_descay_bias=0, gradient_moment=0,
            gradient_moment_bias=0)

        veles_gd_conv.input = formats.Vector(input)
        veles_gd_conv.output = formats.Vector(output)
        veles_gd_conv.weights = formats.Vector(weights)
        veles_gd_conv.bias = formats.Vector(bias)
        veles_gd_conv.err_output = formats.Vector(err_output)

        veles_gd_conv.link_from(workflow.start_point)
        workflow.end_point.unlink_all()
        workflow.end_point.link_from(veles_gd_conv)
        workflow.initialize(device=self.device)

        veles_gd_conv.gradient_bias.map_invalidate()
        veles_gd_conv.gradient_bias.mem[:] = 0

        veles_gd_conv.gradient_weights.map_invalidate()
        veles_gd_conv.gradient_weights.mem[:] = 0

        workflow.run()

        veles_gd_conv.gradient_bias.map_read()
        veles_gd_conv.gradient_weights.map_read()
        veles_gd_conv.err_input.map_read()

        err_input = veles_gd_conv.err_input.mem
        gd_bias = veles_gd_conv.gradient_bias.mem
        gd_weights = veles_gd_conv.gradient_weights.mem

        return err_input, gd_weights, gd_bias

    def test_back_conv(self):
        cudnn_err_input, cudnn_gd_weights, cudnn_gd_bias = \
            self._cudnn_back_conv()

        veles_err_input, veles_gd_weights, veles_gd_bias = \
            self._veles_back_conv()

        logging.info("err_input delta: %f" % self._relative_delta(
            cudnn_err_input, veles_err_input))
        logging.info("gd_weights delta: %f" % self._relative_delta(
            cudnn_gd_weights.reshape(veles_gd_weights.shape),
            veles_gd_weights))
        logging.info("gd_bias delta: %f" % self._relative_delta(cudnn_gd_bias,
                                                                veles_gd_bias))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
