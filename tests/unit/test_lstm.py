# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 3, 2015

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import numpy

from veles.memory import Array
import veles.prng as prng
from veles.tests import AcceleratedTest, assign_backend
from veles.znicz.lstm import LSTM, GDLSTM
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff


class TestLSTM(AcceleratedTest, GDNumDiff):
    ABSTRACT = True

    def setUp(self):
        super(TestLSTM, self).setUp()
        self.precision_threshold = {
            numpy.float64: 1.0e-8,
            numpy.float32: 1.0e-4,
            numpy.float16: 1.0e-2}[self.dtype]

    def test_simple(self):
        self._test(True)

    def _test_extended(self):
        self._test(False)

    def _test(self, simple):
        N = 3
        I = 5
        O = 9

        inp = Array(numpy.zeros((N, I), dtype=self._dtype))
        hid = Array(numpy.zeros((N, O), dtype=self._dtype))
        mem = Array(numpy.zeros((N, O), dtype=self._dtype))
        prng.get().fill(inp.mem)
        prng.get().fill(hid.mem)
        prng.get().fill(mem.mem)

        lstm = LSTM(self.parent, simple=simple,
                    output_sample_shape=hid.shape[1:])
        lstm.input = inp
        lstm.prev_output = hid
        lstm.prev_memory = mem

        lstm.initialize(self.device)

        # Compute LSTM's output manually
        x = numpy.append(inp.mem, hid.mem, axis=1)
        ig = numpy.dot(x, lstm.input_gate.weights.mem.transpose())
        ig += lstm.input_gate.bias.mem
        ig = 1.0 / (1.0 + numpy.exp(-ig))
        mm = numpy.dot(x, lstm.memory_maker.weights.mem.transpose())
        mm += lstm.memory_maker.bias.mem
        mm = 1.7159 * numpy.tanh(0.6666 * mm)
        fg = numpy.dot(x, lstm.forget_gate.weights.mem.transpose())
        fg += lstm.forget_gate.bias.mem
        fg = 1.0 / (1.0 + numpy.exp(-fg))

        im = ig * mm
        fm = fg * mem.mem
        sm = im + fm
        oa = 1.7159 * numpy.tanh(0.6666 * sm)

        if not simple:
            xg = numpy.append(x, sm, axis=1)
        else:
            xg = x
        og = numpy.dot(xg, lstm.output_gate.weights.mem.transpose())

        og += lstm.output_gate.bias.mem
        og = 1.0 / (1.0 + numpy.exp(-og))

        om = oa * og

        vector_value_map = {
            lstm.input: inp.mem.copy(),
            lstm.prev_output: hid.mem.copy(),
            lstm.prev_memory: mem.mem.copy()
        }
        for unit in (lstm.input_gate, lstm.memory_maker, lstm.forget_gate,
                     lstm.output_gate):
            for attr in ("weights", "bias"):
                arr = getattr(unit, attr)
                arr.map_read()
                vector_value_map[arr] = arr.mem.copy()

        lstm.run()
        lstm.output.map_read()
        max_diff = numpy.fabs(lstm.output.mem - om).max()
        self.assertLess(max_diff, self.precision_threshold,
                        "LSTM forward failed")

        target = numpy.zeros((N, O), dtype=self._dtype)
        prng.get().fill(target)
        err_output = Array(lstm.output.mem - target)
        err_memory = Array(numpy.zeros_like(err_output.mem))

        # Backpropagate error manually
        goa = err_output.mem * og
        goa *= oa * oa * (-0.388484177) + 1.14381894

        gog = err_output.mem * oa
        gog *= og * (1.0 - og)
        gogx = numpy.dot(gog, lstm.output_gate.weights.mem)
        if not simple:
            gx = gogx[:, :x.shape[1]].copy()
            goa += gogx[:, x.shape[1]:]
        else:
            gx = gogx.copy()

        gim = goa
        gfm = goa
        gfg = gfm * mem.mem
        gmm = gim * ig
        gig = gim * mm
        gfg *= fg * (1.0 - fg)
        gig *= ig * (1.0 - ig)
        gmm *= mm * mm * (-0.388484177) + 1.14381894
        gx += numpy.dot(gfg, lstm.forget_gate.weights.mem)
        gx += numpy.dot(gmm, lstm.memory_maker.weights.mem)
        gx += numpy.dot(gig, lstm.input_gate.weights.mem)
        ginp = gx[:, :inp.mem.shape[1]]

        gd_lstm = GDLSTM(self.parent, lstm)

        gd_lstm.err_output = err_output
        gd_lstm.err_memory = err_memory

        gd_lstm.initialize(self.device)
        gd_lstm.run()

        gd_lstm.err_input.map_read()
        max_diff = numpy.fabs(gd_lstm.err_input.mem - ginp).max()
        self.assertLess(max_diff, self.precision_threshold,
                        "LSTM backward failed")

        self.info("Checking err_input via numeric differentiation...")
        err_input = gd_lstm.err_input.mem.ravel()
        self.numdiff_check(
            lstm, lstm.input, vector_value_map,
            lstm.output, target, err_input,
            self.info, self.assertLess, GDNumDiff.sse, inp.shape[0])
        self.info("Checked err_input via numeric differentiation: All Ok")


@assign_backend("ocl")
class OCLTestLSTM(TestLSTM):
    pass


@assign_backend("cuda")
class CUDATestLSTM(TestLSTM):
    pass


@assign_backend("numpy")
class NUMPYTestLSTM(TestLSTM):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
