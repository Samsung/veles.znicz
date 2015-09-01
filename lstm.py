# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jul 28, 2015

LSTM unit.

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


from __future__ import division
import weakref

from veles.accelerated_units import AcceleratedWorkflow
from veles.input_joiner import InputJoiner
from veles.znicz.activation import ForwardTanh, BackwardTanh
from veles.znicz.all2all import All2AllSigmoid, All2AllTanh
from veles.znicz.cutter import Cutter1D
from veles.znicz.gd import GDTanh, GDSigmoid
from veles.znicz.multiplier import Multiplier, GDMultiplier
from veles.znicz.nn_units import FullyConnectedOutput
from veles.znicz.summator import Summator


class LSTM(FullyConnectedOutput, AcceleratedWorkflow):
    """LSTM block.

    Must be assigned before initialize():
        input: current input vector
        prev_output: output from the previous LSTM unit (hidden state)
        prev_memory: value of memory cell from the previous LSTM unit

    Updates after run():
        output: current output (hidden state)
        memory: current value of memory cell

    Attributes:
        simple: do not connect memory cell to an output gate.
    """

    MAPPING = {"LSTM"}

    def __init__(self, workflow, **kwargs):
        super(LSTM, self).__init__(workflow, **kwargs)
        self.simple = kwargs.pop("simple", True)

        # Create units
        self.ij = InputJoiner(self)
        self.input_gate = All2AllSigmoid(self, name="input_gate", **kwargs)
        self.forget_gate = All2AllSigmoid(self, name="forget_gate", **kwargs)
        self.memory_maker = All2AllTanh(self, name="memory_maker", **kwargs)

        if not self.simple:
            self.ij_output = InputJoiner(self)
        self.output_gate = All2AllSigmoid(self, name="output_gate", **kwargs)
        self.output_activation = ForwardTanh(self, name="output_activation",
                                             **kwargs)

        self.input_mul = Multiplier(self, name="input_mul")
        self.forget_mul = Multiplier(self, name="forget_mul")
        self.summator = Summator(self, name="memory_cell")
        self.output_mul = Multiplier(self, name="output_mul")

        # Link control flow
        self.ij.link_from(self.start_point)
        self.input_gate.link_from(self.ij)
        self.forget_gate.link_from(self.ij)
        self.memory_maker.link_from(self.ij)
        self.input_mul.link_from(self.input_gate, self.memory_maker)
        self.forget_mul.link_from(self.forget_gate)
        self.summator.link_from(self.input_mul, self.forget_mul)

        if not self.simple:
            self.ij_output.link_from(self.summator, self.ij)
            self.output_gate.link_from(self.ij_output)
        else:
            self.output_gate.link_from(self.ij)

        self.output_activation.link_from(self.summator)
        self.output_mul.link_from(self.output_activation, self.output_gate)
        self.end_point.link_from(self.output_mul)

        # Link unit attributes
        self.ij.link_inputs(self, "input", "prev_output")
        self.input_gate.link_attrs(self.ij, ("input", "output"))
        self.forget_gate.link_attrs(self.ij, ("input", "output"))
        self.memory_maker.link_attrs(self.ij, ("input", "output"))
        self.input_mul.link_attrs(self.input_gate, ("x", "output"))
        self.input_mul.link_attrs(self.memory_maker, ("y", "output"))
        self.forget_mul.link_attrs(self.forget_gate, ("x", "output"))
        self.forget_mul.link_attrs(self, ("y", "prev_memory"))
        self.summator.link_attrs(self.input_mul, ("x", "output"))
        self.summator.link_attrs(self.forget_mul, ("y", "output"))
        self.output_activation.link_attrs(self.summator, ("input", "output"))

        if not self.simple:
            self.ij_output.link_inputs(self.ij, "output")
            self.ij_output.link_inputs(self.summator, "output")
            self.output_gate.link_attrs(self.ij_output, ("input", "output"))
        else:
            self.output_gate.link_attrs(self.ij, ("input", "output"))

        self.output_mul.link_attrs(self.output_gate, ("x", "output"))
        self.output_mul.link_attrs(self.output_activation, ("y", "output"))
        self.link_attrs(self.output_mul, "output")
        self.link_attrs(self.summator, ("memory", "output"))

        self.demand("input", "prev_output", "prev_memory")

    def link_weights(self, src):
        """Links this weights to the weights of src.
        """
        for attr in ("input_gate", "forget_gate", "memory_maker",
                     "output_gate"):
            getattr(self, attr).link_attrs(
                getattr(src, attr), "weights", "bias")


class GDLSTM(AcceleratedWorkflow):
    """Gradient descent unit for LSTM block.

    Must be assigned before initialize():
        err_output: error for backpropagation for output
        err_memory: error for backpropagation for memory cell

    Updates after run():
        err_input: backpropagated error for input
        err_prev_output: error for backpropagation for previous LSTM's output
        err_prev_memory: error for backpropagation for previous LSTM's memory

    Attributes:
        forward: weakref.proxy() from corresponding LSTM instance.
    """
    MAPPING = {"LSTM"}

    def __init__(self, workflow, forward, **kwargs):
        """Constructor.

        Parameters:
            forward: corresponding LSTM instance.
        """
        if forward is None:
            raise ValueError("forward must be provided")
        super(GDLSTM, self).__init__(workflow, **kwargs)

        # Create required gradient units
        self.gd_output_mul = GDMultiplier(self, name="gd_output_mul")
        self.gd_output_activation = BackwardTanh(
            self, name="gd_output_activation")
        self.gd_output_gate = GDSigmoid(self, name="gd_output_gate", **kwargs)
        if not forward.simple:
            self.og_to_summator = Cutter1D(self, name="og_to_summator",
                                           alpha=1, beta=1)
            self.og_to_ij = Cutter1D(self, name="og_to_ij", alpha=1, beta=0)
        self.gd_forget_mul = GDMultiplier(self, name="gd_forget_mul")
        self.gd_input_mul = GDMultiplier(self, name="gd_input_mul")
        self.gd_memory_maker = GDTanh(
            self, name="gd_memory_maker",
            err_input_alpha=1, err_input_beta=1, **kwargs)
        self.gd_forget_gate = GDSigmoid(
            self, name="gd_forget_gate", err_input_alpha=1, err_input_beta=1,
            **kwargs)
        self.gd_input_gate = GDSigmoid(
            self, name="gd_input_gate", err_input_alpha=1, err_input_beta=1,
            **kwargs)
        self.ij_to_input = Cutter1D(self, name="ij_to_input", alpha=1, beta=0)
        self.ij_to_prev_output = Cutter1D(self, name="ij_to_prev_output",
                                          alpha=1, beta=0)

        # Link control flow
        prev = self.gd_output_mul.link_from(self.start_point)
        prev = self.gd_output_activation.link_from(prev)
        prev = self.gd_output_gate.link_from(prev)
        if not forward.simple:
            prev = self.og_to_summator.link_from(prev)
            prev = self.og_to_ij.link_from(prev)
        prev = self.gd_forget_mul.link_from(prev)
        prev = self.gd_input_mul.link_from(prev)
        prev = self.gd_forget_gate.link_from(prev)
        prev = self.gd_memory_maker.link_from(prev)
        prev = self.gd_input_gate.link_from(prev)
        prev = self.ij_to_input.link_from(prev)
        prev = self.ij_to_prev_output.link_from(prev)
        self.end_point.link_from(prev)

        # Link unit attributes
        # gd for output_mul doesn't have weights, so only err_output is needed
        self.gd_output_mul.link_attrs(self, "err_output")
        self.gd_output_mul.link_attrs(forward.output_mul, "x", "y")

        # gd for output_gate has weights, so err_output and weights are needed
        self.gd_output_gate.link_attrs(
            self.gd_output_mul, ("err_output", "err_x"))
        self.gd_output_gate.link_attrs(
            forward.output_gate, "weights", "bias", "input", "output")

        # gd for output activation doesn't have weights,
        # so only err_output is needed
        self.gd_output_activation.link_attrs(
            self.gd_output_mul, ("err_output", "err_y"))
        self.gd_output_activation.link_attrs(
            forward.output_activation, "input", "output")

        # with not simple mode, summator's output is connected to output_gate
        # via input joiner
        if not forward.simple:
            # we need to copy part of the gd_output_gate's err_output
            # to the summator's err_output
            self.og_to_summator.link_attrs(
                self.gd_output_gate, ("input", "err_input"))
            self.og_to_summator.link_attrs(
                forward.ij_output, ("input_offset", "offset_1"),
                ("length", "length_1"))
            # assign err_input for og_to_summator
            self.og_to_summator.link_attrs(
                self.gd_output_activation, ("output", "err_input"))

            # we need to copy part of the gd_output_gate's err_output
            # to the ij's err_output
            self.og_to_ij.link_attrs(
                self.gd_output_gate, ("input", "err_input"))
            self.og_to_ij.link_attrs(
                forward.ij_output, ("input_offset", "offset_0"),
                ("length", "length_0"))
            first = self.og_to_ij
            first_attr = "output"
        else:
            first = self.gd_output_gate
            first_attr = "err_input"

        # forget mul
        self.gd_forget_mul.link_attrs(
            self.gd_output_activation, ("err_output", "err_input"))
        self.gd_forget_mul.link_attrs(
            forward.forget_mul, "x", "y")
        self.link_attrs(self.gd_forget_mul, ("err_prev_memory", "err_y"))

        # forget gate
        self.gd_forget_gate.link_attrs(
            self.gd_forget_mul, ("err_output", "err_x"))
        self.gd_forget_gate.link_attrs(
            forward.forget_gate, "weights", "bias", "input", "output")
        self.gd_forget_gate.link_attrs(first, ("err_input", first_attr))

        # input mul
        self.gd_input_mul.link_attrs(
            self.gd_output_activation, ("err_output", "err_input"))
        self.gd_input_mul.link_attrs(
            forward.input_mul, "x", "y")

        # input gate
        self.gd_input_gate.link_attrs(
            self.gd_input_mul, ("err_output", "err_x"))
        self.gd_input_gate.link_attrs(
            forward.input_gate, "weights", "bias", "input", "output")
        self.gd_input_gate.link_attrs(first, ("err_input", first_attr))

        # memory maker
        self.gd_memory_maker.link_attrs(
            self.gd_input_mul, ("err_output", "err_y"))
        self.gd_memory_maker.link_attrs(
            forward.memory_maker, "weights", "bias", "input", "output")
        self.gd_memory_maker.link_attrs(first, ("err_input", first_attr))

        # to input
        self.ij_to_input.link_attrs(first, ("input", first_attr))
        self.ij_to_input.link_attrs(
            forward.ij, ("input_offset", "offset_0"),
            ("length", "length_0"))
        self.link_attrs(self.ij_to_input, ("err_input", "output"))

        # to prev_output
        self.ij_to_prev_output.link_attrs(first, ("input", first_attr))
        self.ij_to_prev_output.link_attrs(
            forward.ij, ("input_offset", "offset_1"),
            ("length", "length_1"))
        self.link_attrs(self.ij_to_prev_output, ("err_prev_output", "output"))

        self.demand("err_output", "err_memory")

        self.forward = weakref.proxy(forward)
