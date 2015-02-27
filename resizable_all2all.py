import numpy

from veles.znicz.all2all import All2All


class ResizableAll2All(All2All):
    MAPPING = {"all2all_resizable"}

    @All2All.output_sample_shape.setter
    def output_sample_shape(self, value):
        old_neurons_number = self.neurons_number if self.is_initialized else 0
        self._set_output_sample_shape(value)
        if not self.is_initialized:
            return
        if self.neurons_number <= 0:
            raise ValueError(
                "Neurons number must be greater than 0 (got %d)" % value)
        self._adjust_neurons_number(self.neurons_number - old_neurons_number)

    def _adjust_neurons_number(self, delta):
        if not self.weights_transposed:
            old_nn = self.weights.shape[0]
            new_weights = numpy.zeros((old_nn + delta, self.weights.shape[1]),
                                      self.weights.dtype)
            if delta > 0:
                new_weights[:old_nn] = self.weights.mem
                self.fill_array(self.weights_filling, new_weights[old_nn:],
                                self.weights_stddev)
            else:
                new_weights[:] = self.weights.mem[:new_weights.shape[0]]
        else:
            old_nn = self.weights.shape[1]
            new_weights = numpy.zeros((self.weights.shape[0], old_nn + delta),
                                      self.weights.dtype)
            if delta > 0:
                new_weights[:, :old_nn] = self.weights.mem
                self.fill_array(self.weights_filling, new_weights[:, old_nn:],
                                self.weights_stddev)
            else:
                new_weights[:] = self.weights.mem[:, :new_weights.shape[1]]
        self.weights.reset(new_weights)
        self.output.reset()
        self._create_output()
        self.init_vectors(self.weights, self.output)
        self._backend_init_()
