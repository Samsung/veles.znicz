"""
Created on Nov 14, 2013

Gradient descent for convolutional units.

* :class:`GradientDescentConv` couples with :class:`veles.znicz.conv.Conv`
* :class:`GDTanhConv` couples with :class:`veles.znicz.conv.ConvTanh`
* :class:`GDRELUConv` couples with :class:`veles.znicz.conv.ConvRELU`
* :class:`GDStrictRELUConv` couples with \
    :class:`veles.znicz.conv.ConvStrictRELU`

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from __future__ import division

import logging
import numpy
import opencl4py as cl
import scipy.signal
import time
from zope.interface import implementer

from veles.config import root
import veles.error as error
from veles.external.prettytable import PrettyTable
import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.opencl_units import IOpenCLUnit
import veles.znicz.nn_units as nn_units


@implementer(IOpenCLUnit)
class GradientDescentConv(nn_units.GradientDescentBase):
    """Gradient descent for simple convolutional layer (no activation).

    Must be assigned before initialize():
        output
        input
        err_output
        weights
        bias
        batch_size

    Updates after run():
        err_input
        err_output
        weights
        bias

    Creates within initialize():
        err_input

    Attributes:
        krn_err_input_clear_: OpenCL kernel for setting err_input with zeros.
        krn_err_input_: OpenCL kernel for computing err_input.
        krn_weights_: OpenCL kernel for weights update.
        krn_err_output_: OpenCL kernel for err_output update.
        krn_bias_: OpenCL kernel for bias update.
        n_kernels: number of convolutional kernels.
        kx: kernel width.
        ky: kernel height.
    """
    def __init__(self, workflow, **kwargs):
        try:
            n_kernels = kwargs["n_kernels"]
            kx = kwargs["kx"]
            ky = kwargs["ky"]
        except KeyError:
            raise KeyError("n_kernels, kx and ky are required parameters")
        padding = kwargs.get("padding", (0, 0, 0, 0))  # Left Top Right Bottom
        sliding = kwargs.get("sliding", (1, 1))  # X Y
        kwargs["n_kernels"] = n_kernels
        kwargs["kx"] = kx
        kwargs["ky"] = ky
        kwargs["padding"] = padding
        kwargs["sliding"] = sliding
        super(GradientDescentConv, self).__init__(workflow, **kwargs)
        self.n_kernels = n_kernels
        self.kx = kx
        self.ky = ky
        self.padding = tuple(padding)
        self.sliding = tuple(sliding)
        self.cl_const = None
        self.reduce_size = 64

    def init_unpickled(self):
        super(GradientDescentConv, self).init_unpickled()
        self.cl_sources_["gradient_descent_conv.cl"] = {}
        self.krn_err_input_clear_ = None
        self.krn_err_input_ = None
        self.krn_weights_ = None
        self.krn_err_output_ = None
        self.krn_bias_ = None

    def initialize(self, device, **kwargs):
        super(GradientDescentConv, self).initialize(device=device, **kwargs)

        if self.err_output.shape != self.output.shape:
            raise error.BadFormatError("err_output.shape != output.shape")

        batch_size = self.input.mem.shape[0]
        sy = self.input.mem.shape[1]
        sx = self.input.mem.shape[2]
        n_channels = self.input.mem.size // (batch_size * sx * sy)
        n_weights = self.n_kernels * self.kx * self.ky * n_channels

        dtype = self.err_output.mem.dtype

        if self.weights.mem.size != n_weights:
            raise error.BadFormatError(
                "Expected number of weights to match "
                "input, n_kernels, kx, ky parameters")
        if self.include_bias and self.bias.size != self.n_kernels:
            raise error.BadFormatError("Expected bias to match n_kernels")
        if self.input.size != batch_size * sy * sx * n_channels:
            raise error.BadFormatError(
                "Expected input size to match "
                "batch_size * sy * sx * n_channels")

        if (self.need_err_input and (
                self.err_input.mem is None or
                self.err_input.mem.size != self.input.mem.size)):
            self.err_input.reset()
            sh = self.input.mem.shape
            if root.common.unit_test:
                sh = list(sh)
                sh[0] <<= 1
                self.err_input.mem = numpy.zeros(sh, dtype=dtype)
                self.err_input.initialize(self)
                self.err_input.map_write()
                self.err_input.vv = self.err_input.mem
                sh[0] >>= 1
                self.err_input.mem = self.err_input.vv[:sh[0]]
                formats.assert_addr(self.err_input.mem, self.err_input.vv)
                self.err_input.vv[sh[0]:] = numpy.nan
            else:
                self.err_input.mem = numpy.zeros(sh, dtype=dtype)

        if (self.store_gradient and
                (self.gradient_weights.mem is None or
                 self.gradient_weights.mem.size != self.weights.mem.size)):
            self.gradient_weights.reset()
            self.gradient_weights.mem = numpy.zeros_like(self.weights.mem)

        if (self.include_bias and self.store_gradient and
                (self.gradient_bias.mem is None or
                 self.gradient_bias.mem.size != self.bias.mem.size)):
            self.gradient_bias.reset()
            self.gradient_bias.mem = numpy.zeros_like(self.bias.mem)

        self.weights.initialize(self, False)
        if self.include_bias:
            self.bias.initialize(self, False)
        self.output.initialize(self)
        self.input.initialize(self)
        self.err_output.initialize(self)
        self.err_input.initialize(self)
        if self.store_gradient:
            self.gradient_weights.initialize(self, False)
            self.gradient_bias.initialize(self, False)

        if device is not None:
            GradientDescentConv.ocl_init(self, device)

    def ocl_init(self, device):
        batch_size = self.input.mem.shape[0]
        sy = self.input.mem.shape[1]
        sx = self.input.mem.shape[2]
        n_channels = self.input.mem.size // (batch_size * sx * sy)
        dtype = self.err_output.mem.dtype

        self.cl_const = numpy.zeros(5, dtype=dtype)

        side = self.weights.shape[1 if self.weights_transposed else 0]
        other = self.weights.size // side

        if self.factor_ortho:
            if not self.col_sums or self.col_sums.size < other:
                self.col_sums.reset()
                self.col_sums.mem = numpy.zeros(other, dtype=dtype)
            self.col_sums.initialize(self)
        self.reduce_size = formats.roundup(min(self.reduce_size, other), 32)

        defines = {
            'USE_ORTHO': int(bool(self.factor_ortho)),
            'H': other,
            'Y': side,
            'APPLY_GRADIENT': int(self.apply_gradient),
            'WEIGHTS_TRANSPOSED': int(self.weights_transposed),
            'STORE_GRADIENT': int(self.store_gradient),
            'INCLUDE_BIAS': int(self.include_bias),
            'USE_ATOMICS': 1,
            'BATCH': batch_size,
            'SX': sx,
            'SY': sy,
            'N_CHANNELS': n_channels,
            'KX': self.kx,
            'KY': self.ky,
            'N_KERNELS': self.n_kernels,
            'PAD_LEFT': self.padding[0],
            'PAD_TOP': self.padding[1],
            'PAD_RIGHT': self.padding[2],
            'PAD_BOTTOM': self.padding[3],
            'SLIDE_X': self.sliding[0],
            'SLIDE_Y': self.sliding[1],
            'REDUCE_SIZE': self.reduce_size
        }
        self.build_program(defines, "%s/gd_conv_%d_%d.cl" % (
            root.common.cache_dir,
            self.input.mem.size // self.input.mem.shape[0],
            self.output.mem.size // self.output.mem.shape[0]),
            dtype=dtype)

        if self.need_err_input:
            self.krn_err_input_clear_ = self.get_kernel("err_input_clear")
            self.krn_err_input_clear_.set_arg(0, self.err_input.devmem)

            self.krn_err_input_ = self.get_kernel("err_h_update")
            self.krn_err_input_.set_args(self.err_output.devmem,
                                         self.weights.devmem,
                                         self.err_input.devmem)

        self.krn_weights_ = self.get_kernel("weights_update")
        self.krn_weights_.set_args(self.err_output.devmem,
                                   self.input.devmem,
                                   self.weights.devmem,
                                   self.gradient_weights.devmem)

        if self.include_bias:
            self.krn_bias_ = self.get_kernel("bias_update")
            self.krn_bias_.set_args(
                self.err_output.devmem, self.bias.devmem,
                self.gradient_bias.devmem)

        if self.factor_ortho:
            self.krn_compute_col_sums_ = self.get_kernel("compute_col_sums")
            self.krn_compute_col_sums_.set_args(self.weights.devmem,
                                                self.col_sums.devmem)
            self.krn_weights_.set_arg(9, self.col_sums.devmem)

    def gpu_weights_update(self):
        self.input.unmap()
        self.err_output.unmap()
        self.weights.unmap()
        self.gradient_weights.unmap()

        sy = self.input.mem.shape[1]
        sx = self.input.mem.shape[2]
        n_channels = self.input.mem.size // (self.input.mem.shape[0] * sx * sy)

        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.err_output.mem.dtype)]

        if self.factor_ortho:
            self.col_sums.unmap()
            side = self.weights.shape[1 if self.weights_transposed else 0]
            other = self.weights.size // side
            self.execute_kernel(
                [other * self.reduce_size], [self.reduce_size],
                self.krn_compute_col_sums_)

            self.cl_const[4] = self.factor_ortho
            self.krn_weights_.set_arg(8, self.cl_const[4:5])

        self.cl_const[0] = self.learning_rate
        self.cl_const[1] = self.weights_decay
        self.cl_const[2] = self.l1_vs_l2
        self.cl_const[3] = self.gradient_moment
        self.krn_weights_.set_args(
            cl.skip(4), self.cl_const[0:1], self.cl_const[1:2],
            self.cl_const[2:3], self.cl_const[3:4])
        if self.weights_transposed:
            global_size = [
                formats.roundup(self.n_kernels, block_size),
                formats.roundup(self.kx * self.ky * n_channels,
                                block_size)]
        else:
            global_size = [
                formats.roundup(self.kx * self.ky * n_channels,
                                block_size),
                formats.roundup(self.n_kernels, block_size)]
        local_size = [block_size, block_size]
        self.execute_kernel(global_size, local_size, self.krn_weights_)

    def gpu_bias_update(self):
        if not self.include_bias:
            return

        self.err_output.unmap()
        self.bias.unmap()
        self.gradient_bias.unmap()

        self.cl_const[0] = self.learning_rate_bias
        self.cl_const[1] = self.weights_decay_bias
        self.cl_const[2] = self.l1_vs_l2_bias
        self.cl_const[3] = self.gradient_moment_bias
        self.krn_bias_.set_args(
            cl.skip(3), self.cl_const[0:1], self.cl_const[1:2],
            self.cl_const[2:3], self.cl_const[3:4])
        global_size = [self.n_kernels * self.reduce_size]
        local_size = [self.reduce_size]
        self.execute_kernel(global_size, local_size, self.krn_bias_)

    def cpu_weights_update(self):
        # TODO(a.kazantsev): implement in case of transposed weights
        #                    (see OpenCL kernel and just swap the matricies).
        if self.weights_transposed:
            raise NotImplementedError(
                "cpu_run is not implemented for transposed weights")

        self.input.map_read()
        self.err_output.map_read()
        self.weights.map_write()
        self.gradient_weights.map_write()

        dtype = self.weights.mem.dtype
        batch_size = self.current_batch_size
        sy = self.input.mem.shape[1]
        sx = self.input.mem.shape[2]
        n_channels = self.input.mem.size // (self.input.mem.shape[0] * sx * sy)

        sx_full = self.padding[0] + sx + self.padding[2]
        sy_full = self.padding[1] + sy + self.padding[3]
        nx = (sx_full - self.kx) // self.sliding[0] + 1
        ny = (sy_full - self.ky) // self.sliding[1] + 1
        sample_shape = (nx * ny, self.kx * self.ky * n_channels)

        sh = self.err_output.mem.shape
        if len(sh) == 3:
            sh[1] *= sh[2]
            sh[2] = 1
        #  err_output = formats.reshape(self.err_output.mem,
        #                               (sh[0], sh[1] * sh[2], sh[3]))

        # calculate gradient for weights
        gd_weights = numpy.zeros_like(self.weights.mem)
        cut = numpy.empty((self.ky, self.kx, n_channels), dtype=dtype)
        sample = numpy.empty(sample_shape, dtype=dtype)
        for batch in range(batch_size):
            # input data unrolling
            sample = numpy.empty(sample_shape)
            for by, bx in ((by, bx) for by in range(ny) for bx in range(nx)):
                y1, y2 = (by * self.sliding[1],
                          by * self.sliding[1] + self.ky)
                x1, x2 = (bx * self.sliding[0],
                          bx * self.sliding[0] + self.kx)
                i1, i2 = (min(max(y1 - self.padding[1], 0), sy),
                          min(max(y2 - self.padding[1], 0), sy))
                j1, j2 = (min(max(x1 - self.padding[0], 0), sx),
                          min(max(x2 - self.padding[0], 0), sx))
                cut_i1, cut_i2 = (i1 - y1 + self.padding[1],
                                  i2 - y1 + self.padding[1])
                cut_j1, cut_j2 = (j1 - x1 + self.padding[0],
                                  j2 - x1 + self.padding[0])
                cut = numpy.zeros((self.ky, self.kx, n_channels),
                                  dtype=self.input.mem.dtype)
                cut[cut_i1:cut_i2, cut_j1:cut_j2, :] = \
                    self.input.mem[batch, i1:i2, j1:j2, :].reshape(i2 - i1,
                                                                   j2 - j1,
                                                                   n_channels)
                sample[by * nx + bx] = cut.ravel()
            err_out_shape = self.err_output.mem.shape
            out = self.err_output.mem[batch].reshape(err_out_shape[1] *
                                                     err_out_shape[2],
                                                     self.n_kernels)
            gd_weights += numpy.dot(out.transpose(),
                                    sample)

        # update weights
        lr = self.learning_rate
        factor_l12 = self.weights_decay
        l1_vs_l2 = self.l1_vs_l2
        gd_weights_reg = -nn_units.GradientDescentBase.cpu_gradient_step(
            self.weights.mem, gd_weights, lr, factor_l12, l1_vs_l2,
            self.factor_ortho)
        if self.store_gradient:
            gd_weights_reg += self.gradient_weights.mem * self.gradient_moment
            self.gradient_weights.mem[:] = gd_weights_reg[:]
        if self.apply_gradient:
            self.weights.mem += gd_weights_reg

    def cpu_bias_update(self):
        if not self.include_bias:
            return

        self.err_output.map_read()
        self.bias.map_write()
        self.gradient_bias.map_write()

        batch_size = self.current_batch_size
        err_out_shape = self.err_output.mem.shape

        # calculate gradient for bias
        gd_bias = numpy.zeros_like(self.bias.mem)
        for batch in range(batch_size):
            out = self.err_output.mem[batch].reshape(err_out_shape[1] *
                                                     err_out_shape[2],
                                                     self.n_kernels)
            gd_bias += numpy.add.reduce(out)
        # update bias
        lr = self.learning_rate
        factor_l12 = self.weights_decay
        l1_vs_l2 = self.l1_vs_l2
        gd_bias_reg = -nn_units.GradientDescentBase.cpu_gradient_step(
            self.bias.mem, gd_bias, lr, factor_l12, l1_vs_l2)
        if self.store_gradient:
            gd_bias_reg += self.gradient_bias.mem * self.gradient_moment_bias
            self.gradient_bias.mem[:] = gd_bias_reg[:]
        if self.apply_gradient:
            self.bias.mem += gd_bias_reg

    def gpu_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return
        self.err_input.unmap()
        self.err_output.unmap()
        self.weights.unmap()

        # Clear the resulting matrix
        self.execute_kernel([self.err_input.mem.size], None,
                            self.krn_err_input_clear_)

        batch_size = self.input.mem.shape[0]
        sy = self.input.mem.shape[1]
        sx = self.input.mem.shape[2]
        n_channels = self.input.mem.size // (batch_size * sx * sy)
        block_size = self.device.device_info.BLOCK_SIZE[
            opencl_types.numpy_dtype_to_opencl(self.err_output.mem.dtype)]
        kernel_size = self.kx * self.ky * n_channels
        global_size = [
            formats.roundup(kernel_size, block_size),
            formats.roundup(
                batch_size *
                ((sx + self.padding[0] + self.padding[2] - self.kx) //
                 self.sliding[0] + 1) *
                ((sy + self.padding[1] + self.padding[3] - self.ky) //
                 self.sliding[1] + 1),
                block_size)]
        local_size = [block_size, block_size]
        self.execute_kernel(global_size, local_size, self.krn_err_input_)

    def cpu_err_input_update(self):
        """Backpropagate error (will compute err_input).
        """
        if not self.need_err_input:
            return
        if self.weights_transposed:
            raise NotImplementedError(
                "cpu_run is not implemented for transposed weights")
        self.err_input.map_invalidate()
        self.err_output.map_read()
        self.weights.map_read()

        batch_size = self.input.mem.shape[0]
        sy = self.input.mem.shape[1]
        sx = self.input.mem.shape[2]
        n_channels = self.input.mem.size // (batch_size * sx * sy)
        sx_full = self.padding[0] + sx + self.padding[2]
        sy_full = self.padding[1] + sy + self.padding[3]

        self.err_input.mem[:] = 0
        # initialize sparse output error
        sparse_err_output = numpy.zeros((
            batch_size, sy_full - self.ky + 1, sx_full - self.kx + 1,
            self.n_kernels), dtype=self.err_output.mem.dtype)
        for (batch, i, j, k), err in numpy.ndenumerate(self.err_output.mem):
            sparse_err_output[batch, i * self.sliding[1],
                              j * self.sliding[0], k] = err
        err_sample = numpy.empty((sy_full - self.ky + 1,
                                  sx_full - self.kx + 1))
        for batch, k in ((batch, k)
                         for batch in range(batch_size)
                         for k in range(self.n_kernels)):
            err_sample[:] = sparse_err_output[batch, :, :, k]
            cur_kernel = self.weights.mem[k].reshape(self.ky, self.kx,
                                                     n_channels)
            for ch in range(n_channels):
                err_input_full = scipy.signal.convolve2d(err_sample,
                                                         cur_kernel[:, :, ch],
                                                         mode='full')
                self.err_input.mem[batch, :, :, ch] += \
                    err_input_full[self.padding[1]:(sy_full - self.padding[3]),
                                   self.padding[0]:(sx_full - self.padding[2])]

    def print_debug_data(self, t_start):
        """
        Show weights statistics
        """
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        self.weights.map_read()
        self.bias.map_read()
        self.gradient_bias.map_read()
        self.gradient_weights.map_read()

        weights = self.weights.mem
        bias = self.bias.mem
        grad_weight = self.gradient_weights.mem
        grad_bias = self.gradient_bias.mem

        n_input = self.input.mem.size // self.input.mem.shape[0]
        n_output = self.output.mem.size // self.output.mem.shape[0]
        delta_time = time.time() - t_start

        stats_table = PrettyTable("n_input", "n_output", "time")
        stats_table.float_format = ".3"
        stats_table.add_row(n_input, n_output, delta_time)
        self.debug("\n" + stats_table.get_string())

        weight_table = PrettyTable("TYPE", "Mean", "StdDev", "Min", "Max")
        weight_table.float_format = ".10"

        actions = [("Weight", weights), ("Bias", bias)]
        if self.store_gradient:
            actions += [("Grad Weight", grad_weight), ("Grad Bias", grad_bias)]

        for (w_name, w_array) in actions:
            w_mean = numpy.mean(w_array)
            w_stddev = numpy.std(w_array)
            w_min = numpy.min(w_array)
            w_max = numpy.max(w_array)
            weight_table.add_row(w_name, w_mean, w_stddev, w_min, w_max)
        self.debug("\n" + weight_table.get_string())

    def gpu_err_output_update(self):
        """Multiply err_output by activation derivative by output.
        """
        if self.krn_err_output_ is None:
            return
        self.output.unmap()
        self.err_output.unmap()
        self.execute_kernel([self.err_output.mem.size], None,
                            self.krn_err_output_)

    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative by output.
        """
        pass

    def ocl_run(self):
        """Do gradient descent.
        """
        t1 = time.time()
        self.gpu_err_output_update()
        self.gpu_err_input_update()
        self.gpu_weights_update()
        self.gpu_bias_update()
        self.print_debug_data(t1)

    def cpu_run(self):
        t1 = time.time()
        self.cpu_err_output_update()
        self.cpu_err_input_update()
        self.cpu_weights_update()
        self.cpu_bias_update()
        self.print_debug_data(t1)


class GDTanhConv(GradientDescentConv):
    """Gradient Descent for f(x) = 1.7159 * tanh(0.6666 * s), s = (W * x + b),
       y = a * tanh(b * s).

    f'(s) = (a * tanh(b * s))' = a * tanh'(b * s) * b
          = a * (1.0 - tanh^2(b * s)) * b
          = a * b - a * b * tanh^2(b * s)
          = a * b - y * y * b / a
          = y * y * (-b / a) + (a * b)
          = y * y * (-0.388484177) + 1.14381894
    """
    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative by s
           in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= output * output * (-0.388484177) + 1.14381894

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_tanh.cl"] = {}
        super(GDTanhConv, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_args(self.err_output.devmem,
                                      self.output.devmem)


class GDRELUConv(GradientDescentConv):
    """Gradient Descent for f(x) = log(1.0 + exp(s)), s = (W * x + b),
       y = log(1.0 + exp(s)).

    f'(s) = 1.0 / (1.0 + exp(-s)) = 1.0 - exp(-y)
    """
    def cpu_err_output_update(self):
        """Multiply err_output by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= 1.0 - numpy.exp(-output)

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_relu.cl"] = {}
        super(GDRELUConv, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_args(self.err_output.devmem,
                                      self.output.devmem)


class GDStrictRELUConv(GradientDescentConv):
    """Gradient Descent for strict ReLU (like in CAFFE)

    :math:`f(x) = \\max(x, 0)`

    :math:`f'(s) = \\begin{cases}1 & s > 0 \\\\ 0 & else. \\\\ \\end{cases}`
    """
    def cpu_err_output_update(self):
        """Multiply `err_output` by activation derivative by s
        in terms of output.
        """
        self.output.map_read()
        self.err_output.map_write()
        output = self.output.mem
        self.err_output.mem *= numpy.greater(output, 0)

    def initialize(self, device, **kwargs):
        self.cl_sources_["gradient_descent_strict_relu.cl"] = {}
        super(GDStrictRELUConv, self).initialize(device=device, **kwargs)
        if self.device is None:
            return
        self.krn_err_output_ = self.get_kernel("err_y_update")
        self.krn_err_output_.set_args(self.err_output.devmem,
                                      self.output.devmem)
