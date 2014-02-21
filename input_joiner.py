"""
Created on Oct 29, 2013

Joins several inpus into one continuous output.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import numpy
import config
import error
import formats
import opencl_types
import units
import znicz_config


class InputJoiner(units.OpenCLUnit):
    """Joins several minibatch inputs into one continuous minibatch output.

    Should be assigned before initialize():
        inputs

    Updates after run():
        output

    Creates within initialize():
        output

    Attributes:
        inputs: list of inputs of type formats.Vector().
        output: formats.Vector().
        output_sample_shape: shape of an output sample, if None,
                             will be a plain sum of input sample shapes.
        minibatch_size: size of the minibatch (will be set to the minimum
                        of the first shapes from the inputs
                        if not provided prior to the initialize)
    """
    def __init__(self, workflow, **kwargs):
        output_sample_shape = kwargs.get("output_sample_shape")
        inputs = kwargs.get("inputs")
        kwargs["output_sample_shape"] = output_sample_shape
        kwargs["inputs"] = inputs
        super(InputJoiner, self).__init__(workflow, **kwargs)
        self.inputs = [] if inputs == None else inputs
        self.output = formats.Vector()
        self.output_sample_shape = output_sample_shape
        self.cl_const = numpy.zeros(4, dtype=numpy.int32)
        self.minibatch_size = [None]

    def init_unpickled(self):
        super(InputJoiner, self).init_unpickled()
        self.cl_sources_["join.cl"] = {}
        self.krn_ = None

    def initialize(self):
        if not len(self.inputs):
            raise error.ErrBadFormat("inputs should not be empty")

        super(InputJoiner, self).initialize()

        if self.minibatch_size[0] == None:
            minibatch_size = self.inputs[0].v.shape[0]
            for i in range(1, len(self.inputs)):
                minibatch_size = min(minibatch_size, self.inputs[i].v.shape[0])
            self.minibatch_size[0] = minibatch_size
        else:
            minibatch_size = self.minibatch_size[0]

        if self.output_sample_shape == None:
            self.output_sample_shape = [0]
            for inp in self.inputs:
                if inp.v == None:
                    raise error.ErrBadFormat(
                        "output_sample_shape should be provided "
                        "if any of the inputs was not initialized "
                        "before this point")
                self.output_sample_shape[0] += inp.v.size // inp.v.shape[0]

        sh = [minibatch_size]
        sh.extend(self.output_sample_shape)
        if (self.output.v == None or self.output.v.size != numpy.prod(sh)):
            self.output.reset()
            self.output.v = numpy.zeros(sh, dtype=self.inputs[0].v.dtype)
        else:
            self.output.v = formats.reshape(self.output.v, sh)

        self.output.initialize(self.device)

        if self.device == None:
            return

        if self.krn_ == None:
            defines = {
                'etype': opencl_types.numpy_dtype_to_opencl(
                                            self.output.v.dtype)
            }
            self.build_program(defines, "%s/join_%s.cl" % (config.cache_dir,
                "_".join(str(x) for x in self.output_sample_shape)))

            self.krn_ = self.get_kernel("join2")
            self.krn_.set_arg(0, self.output.v_)

    def cpu_run(self):
        self.output.map_invalidate()  # we will update output on CPU
        minibatch_size = self.minibatch_size[0]
        low = 0
        output_sample_size = self.output.v.size // self.output.v.shape[0]
        for inp in self.inputs:
            inp.map_read()
            high = min(low + inp.v.size // inp.v.shape[0], output_sample_size)
            if low >= high:
                break
            self.output.v[:minibatch_size, low:high] = (
                inp[:minibatch_size, :high - low])
            low = high

    def gpu_run(self):
        self.output.unmap()  # we will update output on GPU
        minibatch_size = self.minibatch_size[0]
        low = 0
        output_sample_size = self.output.v.size // self.output.v.shape[0]
        self.cl_const[3] = output_sample_size
        self.krn_.set_arg(6, self.cl_const[3])
        a = None
        a_size = 0
        b = None
        for inp in self.inputs:
            inp.unmap()  # we will use input on GPU
            if a == None:
                a = inp
                a_size = a.v.size // a.v.shape[0]
                continue
            b = inp
            b_size = b.v.size // b.v.shape[0]
            high = min(low + a_size + b_size,
                       output_sample_size)
            if low >= high:
                break
            self.cl_const[0] = a_size
            self.cl_const[1] = b_size
            self.cl_const[2] = low
            self.krn_.set_arg(1, a.v_)
            self.krn_.set_arg(2, b.v_)
            self.krn_.set_arg(3, self.cl_const[0])
            self.krn_.set_arg(4, self.cl_const[1])
            self.krn_.set_arg(5, self.cl_const[2])
            global_size = [high - low, minibatch_size]
            event = self.enqueue_nd_range_kernel(self.krn_,
                                                 global_size, None)
            event.wait()
            low = high
            a = None
            a_size = 0
            b = None
        if a != None:
            b_size = (b.v.size // b.v.shape[0] if b != None else 0)
            high = min(low + a_size + b_size,
                       output_sample_size)
            if low < high:
                self.cl_const[0] = a_size
                self.cl_const[1] = b_size
                self.cl_const[2] = low
                self.krn_.set_arg(1, a.v_)
                self.krn_.set_arg(2, b.v_ if b != None else None)
                self.krn_.set_arg(3, self.cl_const[0])
                self.krn_.set_arg(4, self.cl_const[1])
                self.krn_.set_arg(5, self.cl_const[2])
                global_size = [high - low, minibatch_size]
                event = self.enqueue_nd_range_kernel(self.krn_,
                                                     global_size, None)
                event.wait()
