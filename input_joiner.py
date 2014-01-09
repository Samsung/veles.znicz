"""
Created on Oct 29, 2013

Joins several inpus into one continuous output.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import units
import formats
import numpy
import error
import config
import znicz_config
import pyopencl


class InputJoiner(units.OpenCLUnit):
    """Joins several inpus into one continuous output.

    Should be assigned before initialize():
        inputs

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        inputs: list of inputs of type formats.Vector().
        output: formats.Vector().
        output_shape: shape of an output, if None, will be plain sum of inputs.
    """
    def __init__(self, output_shape=None, inputs=None, device=None):
        super(InputJoiner, self).__init__(device)
        self.inputs = [] if inputs == None else inputs
        self.output = formats.Vector()
        self.output_shape = output_shape
        self.cl_const = numpy.zeros(2, dtype=numpy.int32)

    def init_unpickled(self):
        super(InputJoiner, self).init_unpickled()
        self.cl_sources_["join.cl"] = ""
        self.krn_ = None

    def initialize(self):
        if not len(self.inputs):
            raise error.ErrBadFormat("inputs should not be empty")

        super(InputJoiner, self).initialize()

        if self.output_shape == None:
            self.output_shape = [0]
            for inp in self.inputs:
                if inp.v == None:
                    raise error.ErrBadFormat("output_shape should be provided "
                        "if any of the inputs was not initialized "
                        "before this point")
                self.output_shape[0] += inp.v.size

        if (self.output.v == None or
            self.output.v.size != numpy.prod(self.output_shape)):
            self.output.reset()
            self.output.v = numpy.zeros(self.output_shape,
                                        dtype=self.inputs[0].v.dtype)
        else:
            self.output.v = formats.reshape(self.output.v, self.output_shape)

        self.output.initialize(self.device)

        if self.device == None:
            return

        if self.krn_ == None:
            defines = ("%s\n"
                       "#define etype %s\n\n" % (
                       config.cl_defines[config.c_dtype],
                       config.numpy_dtype_to_opencl(self.output.v.dtype)))
            self.build_program(defines, "%s/join_%s.cl" % (config.cache_dir,
                "_".join(str(x) for x in self.output_shape)))

            self.krn_ = pyopencl.Kernel(self.prg_, "join2")
            self.krn_.set_arg(0, self.output.v_)

    def cpu_run(self):
        self.output.map_invalidate()  # we will update output on CPU
        low = 0
        output_size = self.output.v.size
        for inp in self.inputs:
            inp.map_read()
            high = min(low + inp.v.size, output_size)
            if low >= high:
                break
            self.output.v[low:high] = inp[:high - low]
            low = high

    def gpu_run(self):
        self.output.unmap()  # we will update output on GPU
        low = 0
        output_size = self.output.v.size
        a = None
        a_size = 0
        b = None
        for inp in self.inputs:
            inp.unmap()  # we will use input on GPU
            if a == None:
                a = inp
                a_size = a.v.size
                continue
            b = inp
            high = min(low + a_size + b.v.size, output_size)
            if low >= high:
                break
            self.cl_const[0] = low
            self.cl_const[1] = a_size
            self.krn_.set_arg(1, a.v_)
            self.krn_.set_arg(2, b.v_)
            self.krn_.set_arg(3, self.cl_const[0])
            self.krn_.set_arg(4, self.cl_const[1])
            global_size = [high - low]
            event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                self.krn_, global_size, None)
            event.wait()
            low = high
            a = None
            a_size = 0
            b = None
        if a != None:
            high = min(low + a_size + (b.v.size if b != None else 0),
                       output_size)
            if low < high:
                self.cl_const[0] = low
                self.cl_const[1] = a_size
                self.krn_.set_arg(1, a.v_)
                self.krn_.set_arg(2, b.v_ if b != None else None)
                self.krn_.set_arg(3, self.cl_const[0])
                self.krn_.set_arg(4, self.cl_const[1])
                global_size = [high - low]
                event = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                    self.krn_, global_size, None)
                event.wait()
