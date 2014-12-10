import libcudnn as cudnn
import logging
import numpy
import pycuda.driver as cuda
import unittest
from zope.interface import implementer

import veles.memory as formats
from veles import error
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
