"""
Created on May 26, 2014

Helper class for numeric differentiation tests.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import veles.formats as formats


class GDNumDiff(object):
    """Helper class for numeric differentiation tests.
    """
    def numdiff_check(self, forward, vector_to_check, vector_value_map,
                      vector_output, target, derivative_to_check,
                      logging_info, assertLess):
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
            if v == vector_to_check:
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
                    k.map_invalidate()
                    formats.ravel(k.mem)[:] = v.ravel()[:]
                mem[offs] = mem[offs] + p
                forward.run()
                vector_output.map_read()
                numdiff.errs[i] = (numpy.square(vector_output.mem.ravel() -
                                                target.ravel()).sum() * 0.5)
            derivative = numdiff.derivative
            d = numpy.fabs(derivative - derivative_to_check[offs])
            logging_info("%.2f %.2f %.2f" % (derivative,
                                             derivative_to_check[offs], d))
            assertLess(d, 0.05, "Numeric diff test failed")

    def numdiff_check_gd(self, forward, inp, weights, bias, target,
                         err_input, weights_derivative, bias_derivative,
                         logging_info, assertLess):
        """Tests all derivatives for a typical gradient descent unit.
        """
        for vector, derivative in ((forward.input, err_input),
                                   (forward.weights, weights_derivative),
                                   (forward.bias, bias_derivative)):
            self.numdiff_check(
                forward, vector, {forward.input: inp,
                                  forward.weights: weights,
                                  forward.bias: bias},
                forward.output, target, derivative,
                logging_info, assertLess)
