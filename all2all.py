"""
Created on Mar 20, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import numpy
import sys


class All2All(filters.GeneralFilter):
    """All2All layer to layer
    
    Attributes:
        output_layer_size: size of the output layer
        weights_amplitude: amplitude of the default random distribution of weights
        rand: numpy-style random generator function
        weights: weights
        bias: bias weights
        mtime: modification time of an input
    """
    def __init__(self, parent, output_layer_size, weights_amplitude = 0.05, rand = numpy.random.rand):
        super(All2All, self).__init__(parent)
        self.output_layer_size = output_layer_size
        self.weights_amplitude = weights_amplitude
        self.rand = rand
        self.weights = None
        self.bias = None
        self.mtime = 0

    def feed_from_batch(self, src):
        """Forward propagation from batch. 
        """
        input = src.output.data
        if not self.weights:
            n = input.size // input.shape[0] * self.output_layer_size
            #print("Weights count: %d" % (n))
            self.weights = self.rand(n).astype(numpy.float32)
            self.weights *= 2.0 * self.weights_amplitude
            self.weights -= self.weights_amplitude
            #print("Weights range: [%.3f, %.3f]" % (self.weights.min(), self.weights.max()))
            self.bias = self.rand(self.output_layer_size).astype(numpy.float32)
            self.bias *= 2.0 * self.weights_amplitude
            self.bias -= self.weights_amplitude

        #TODO(a.kazantsev): Check src.output for OpenCL objects and use them if it has any.
        

        #TODO(a.kazantsev): notify parent on completion (OpenCL event)
        #if self.parent:
        #    self.parent.output_changed(self)
        return

    def input_changed(self, src):
        """GeneralFilter method
        """
        if self.mtime >= src.output.mtime:
            return
        self.mtime = src.output.mtime
        if src.output.__class__.__name__ == "DataBatch":
            return self.feed_from_batch(src)
        return 
