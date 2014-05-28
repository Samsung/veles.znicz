"""
Created on May 26, 2014

Helper class for numeric differentiation tests.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import veles.formats as formats
import veles.znicz.all2all as all2all


class GDNumDiff(object):
    """Helper class for numeric differentiation tests.
    """
    def numdiff_check(self, forward, vector_to_check, vector_value_map,
                      vector_output, target, derivative_to_check,
                      logging_info, assertLess, error_function):
        """Checks derivative by numeric differentiation based on MSE to target.

        Parameters:
            forward: forward unit instance.
            vector_to_check: vector on which to do numeric differentiation.
            vector_value_map: dictionary of vectors to set => its values,
                              should contain vector_to_check.
            vector_output: output vector to compute MSE based on target.
            target: target numpy array for MSE criteria.
            derivative: numpy array of derivative to check
                        (computed by backward unit).
            logging_info: pointer to logging function
                          (logging.info for example).
            assertLess: pointer to assertLess function.
        """
        for k, v in forward.__dict__.items():
            if id(v) == id(vector_to_check):
                nme = k
                break
        else:
            nme = str(vector_to_check)
        logging_info("Checking %s.%s with numeric differentiation",
                     forward.__class__.__name__, nme)

        numdiff = formats.NumDiff()

        mem = formats.ravel(vector_to_check.mem)
        derivative_to_check = derivative_to_check.ravel()
        for offs in range(mem.shape[0]):
            for i, p in enumerate(numdiff.points):
                for k, v in vector_value_map.items():
                    if v is None or k is None or k.mem is None:
                        continue
                    k.map_invalidate()
                    formats.ravel(k.mem)[:] = v.ravel()[:]
                mem[offs] = mem[offs] + p
                forward.run()
                vector_output.map_read()
                numdiff.errs[i] = error_function(vector_output.mem.ravel(),
                                                 target.ravel())
            derivative = numdiff.derivative
            d = numpy.fabs(derivative - derivative_to_check[offs])
            logging_info("%.2e %.2e %.2e" %
                (derivative, derivative_to_check[offs], d))
            assertLess(d, 0.01, "Numeric diff test failed")

    @staticmethod
    def mse(y, t):
        return numpy.square(y - t).sum() * 0.5

    @staticmethod
    def cross_entropy(y, t):
        idx = numpy.nonzero(t)
        return (t[idx] * numpy.log(t[idx] / y[idx])).sum()

    def numdiff_check_gd(self, forward, inp, weights, bias, target,
                         err_input, weights_derivative, bias_derivative,
                         logging_info, assertLess):
        """Tests all derivatives for a typical gradient descent unit.
        """
        for vector, derivative in ((forward.input, err_input),
                                   (forward.weights, weights_derivative),
                                   (forward.bias, bias_derivative)):
            if derivative is None:
                continue
            self.numdiff_check(
                forward, vector, {forward.input: inp,
                                  forward.weights: weights,
                                  forward.bias: bias},
                forward.output, target, derivative,
                logging_info, assertLess,
                GDNumDiff.cross_entropy
                if isinstance(forward, all2all.All2AllSoftmax)
                else GDNumDiff.mse)
