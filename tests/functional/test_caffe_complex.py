# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 06, 2014

A complex test: comparison with CAFFE on full CIFAR model

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

import os
import tarfile

import numpy

from veles.compat import IntEnum
from veles.memory import Array
from veles.tests import AcceleratedTest
from veles.znicz import (activation, all2all, conv, evaluator, pooling,
                         normalization)
from veles.znicz.tests.functional.gradient_unit_factory import \
    GradientUnitFactory
from veles.znicz.tests.functional import StandardTest
from veles.znicz.tests.functional.test_caffe import CaffeTestBase


class PropType(IntEnum):
    """
    Propagation: forward of backward
    """
    forward = 0
    backward = 1


class WhenTaken(IntEnum):
    """
    When snapshot was taken: before of after propagation
    """
    before = 0
    after = 1


class LayerInfo(object):
    """
    CAFFE snapshot info class
    """
    def __init__(self, name=None, l_type=None, prop=None, when_taken=None,
                 time=None):
        self.name = name
        self.layer_type = l_type
        self.propagation = prop
        self.time = time
        self.when_taken = when_taken
        self.path = None


class ComplexTest(CaffeTestBase):
    def setUp(self):
        super(ComplexTest, self).setUp()
        self.layer_dict = {}

    def getParent(self):
        return AcceleratedTest.getParent(self)

    def _compress_snapshots(self):
        out_archive = tarfile.open(name=os.path.join(
            self.data_dir_path, "cifar_export.tar.xz"), mode='w:xz')
        out_archive.add(name=os.path.join(
            self.data_dir_path, "cifar_export"), arcname="cifar_export")
        out_archive.close()

    def _extract_snapshots(self):
        in_archive = tarfile.open(name=os.path.join(
            self.data_dir_path, "cifar_export.tar.xz"), mode="r:xz")
        in_archive.extractall(self.data_dir_path)

    def _prepare_snapshots(self):
        """
        Reads a CIFAR export directory. Fills layer info path dictionary.
        """
        # self._compress_snapshots()
        in_dir = os.path.join(self.data_dir_path, "cifar_export")
        if not os.path.exists(in_dir):
            self.info("Extracting snapshots...")
            self._extract_snapshots()
            self.info("Finished")

        iters = os.listdir(in_dir)
        for _iter in iters:
            iter_dir = os.path.join(in_dir, _iter)
            filenames = os.listdir(iter_dir)
            layer_infos = []
            for name in filenames:
                nibbles = name.split(".")

                info = LayerInfo(nibbles[1], nibbles[0], PropType[nibbles[2]],
                                 WhenTaken[nibbles[3]], time=int(nibbles[4]))
                info.path = os.path.join(iter_dir, name)

                layer_infos.append(info)
            layer_infos.sort(key=lambda x: x.time)
            iter_dict = {}
            self.layer_dict[int(_iter)] = iter_dict
            for info in layer_infos:
                if info.name not in iter_dict:
                    iter_dict[info.name] = {}
                this_layer_dict = iter_dict[info.name]
                if info.propagation not in this_layer_dict:
                    this_layer_dict[info.propagation] = {}
                this_prop_dict = this_layer_dict[info.propagation]
                this_prop_dict[info.when_taken] = info.path

    def _load_blob(self, layer_name, prop, when_taken, blob_name, iteration):
        """
        Loads blob by its name and parameters.

        Args:
            layer_name(str): layer name in CAFFE
            prop(:class:`PropType`) type of propagation: fwd or backward
            when_taken(:class:`WhenTaken`): was snapshot taken before or after
            blob_name(std): blob name in snapshot file
            iteration(int): iteration, from which the snapshot was taken

        Returns:
            :class:`numpy.ndarray(dtype=float64)`
        """
        snapshot_path = self._snapshot_path(layer_name, prop, when_taken,
                                            iteration)
        lines = self._read_lines_by_abspath(snapshot_path)
        return self._read_array(blob_name, lines)

    def _snapshot_path(self, layer_name, prop_type, when_taken, iteration=2):
        return self.layer_dict[iteration][layer_name][prop_type][when_taken]

    def _diff(self, array1, array2):
        """
        Calculates a difference between two ndarrays

        Args:
            array1(:class:`numpy.ndarray`):
            array2(:class:`numpy.ndarray`):

        Returns:
            float
        """
        assert array1.shape == array2.shape
        return (numpy.sum(numpy.abs(array1 - array2)) /
                numpy.sum(numpy.abs(array1)) * 100)
#        a = array1.copy()
#        b = array2.copy()
#        return numpy.fabs(a - b).max() / numpy.fabs(a).max()

    def _create_fwd_units(self, iteration):
        # Layer 1: CONV+POOL
        conv1 = conv.Conv(self.parent, name="conv1",
                          kx=5, ky=5, padding=(2, 2, 2, 2),
                          sliding=(1, 1), n_kernels=32,
                          weights_filling="gaussian", weights_stddev=10 ** -4,
                          bias_filling="constant", bias_stddev=0)
        conv1_bottom = self._load_blob(
            "conv1", PropType.forward, WhenTaken.before, "bottom_0", iteration)
        conv1.input = Array(conv1_bottom)
        conv1.link_from(self.parent.start_point)

        pool1 = pooling.MaxPooling(self.parent, name="pool1",
                                   kx=3, ky=3, sliding=(2, 2))
        pool1.link_from(conv1)
        pool1.link_attrs(conv1, ("input", "output"))

        relu1 = activation.ForwardStrictRELU(self.parent, name="relu1")
        relu1.link_from(pool1)
        relu1.link_attrs(pool1, ("input", "output"))

        norm1 = normalization.LRNormalizerForward(
            self.parent, name="norm1", n=3, alpha=0.00005,
            beta=0.75, k=1)
        norm1.link_from(relu1)
        norm1.link_attrs(relu1, ("input", "output"))

        # Layer 2: CONV+POOL
        conv2 = conv.Conv(self.parent, name="conv2",
                          kx=5, ky=5, padding=(2, 2, 2, 2),
                          sliding=(1, 1), n_kernels=32,
                          weights_filling="gaussian", weights_stddev=10 ** -2,
                          bias_filling="constant", bias_stddev=0)

        conv2.link_from(norm1)
#        conv2_bottom = self._load_blob("conv2", PropType.forward,
#                                       WhenTaken.before, "bottom_0")
#        conv2.input = Array(conv2_bottom)
        conv2.link_attrs(norm1, ("input", "output"))

        relu2 = activation.ForwardStrictRELU(self.parent, name="relu2")
        relu2.link_from(conv2)
        relu2.link_attrs(conv2, ("input", "output"))

        pool2 = pooling.AvgPooling(self.parent, name="pool2",
                                   kx=3, ky=3, sliding=(2, 2))
        pool2.link_from(relu2)
        pool2.link_attrs(relu2, ("input", "output"))

        norm2 = normalization.LRNormalizerForward(
            self.parent, name="norm2", n=3, alpha=0.00005,
            beta=0.75, k=1)
        norm2.link_from(pool2)
        norm2.link_attrs(pool2, ("input", "output"))

        # Layer 3: CONV+POOL
        conv3 = conv.Conv(self.parent, name="conv3",
                          kx=5, ky=5, padding=(2, 2, 2, 2),
                          sliding=(1, 1), n_kernels=64,
                          weights_filling="gaussian", weights_stddev=10 ** -2,
                          bias_filling="constant", bias_stddev=0)
        conv3.link_from(norm2)
        conv3.link_attrs(norm2, ("input", "output"))

        relu3 = activation.ForwardStrictRELU(self.parent, name="relu3")
        relu3.link_from(conv3)
        relu3.link_attrs(conv3, ("input", "output"))

        pool3 = pooling.AvgPooling(self.parent, name="pool3",
                                   kx=3, ky=3, sliding=(2, 2))
        pool3.link_from(relu3)
        pool3.link_attrs(relu3, ("input", "output"))

        # Layer 4: FC
        ip_sm = all2all.All2AllSoftmax(
            self.parent, name="ip1", output_sample_shape=10,
            weights_filling="uniform", weights_stddev=10 ** -2,
            bias_filling="constant", bias_stddev=0)
        ip_sm.link_from(pool3)
        ip_sm.link_attrs(pool3, ("input", "output"))
        ########

    def _load_labels_and_data(self, _iter):
        labels = self._load_blob(
            "loss", PropType.backward, WhenTaken.before,
            "bottom_1", _iter).astype(numpy.int32).ravel()

        ev = self.parent["ev"]
        if ev.labels is not None:
            ev.labels.map_write()
            ev.labels.mem[:] = labels[:]
        else:
            ev.labels = Array(labels)

        conv1_bottom = self._load_blob(
            "conv1", PropType.forward, WhenTaken.before, "bottom_0", _iter)
        conv1 = self.parent["conv1"]
        if conv1.input is None:
            conv1.input = Array(conv1_bottom)
        else:
            conv1.input.map_write()
            conv1.input.mem[:] = conv1_bottom[:]

    def _create_gd_units(self, _iter):
        # BACKPROP
        # BACK LAYER 4: EV + GD_A2ASM

        ip_sm = self.parent["ip1"]

        ev = evaluator.EvaluatorSoftmax(self.parent, name="ev")
        self._load_labels_and_data(_iter)
        ev.link_from(ip_sm)
        ev.batch_size = self.n_pics
        ev.link_attrs(ip_sm, "max_idx")
        ev.link_attrs(ip_sm, "output")

        gd_ip_sm = GradientUnitFactory.create(
            ip_sm, name="gd_ip1",
            learning_rate=0.001, learning_rate_bias=0.002,
            weights_decay=1.0, weights_decay_bias=0.0,
            gradient_moment=0.9, gradient_moment_bias=0.9)

        gd_ip_sm.link_from(ev)
        gd_ip_sm.link_attrs(ev, "err_output")

        # BACK LAYER 3: CONV
        gd_pool3 = GradientUnitFactory.create(
            self.parent["pool3"], "gd_pool3")
        gd_pool3.link_from(gd_ip_sm)
        gd_pool3.link_attrs(gd_ip_sm, ("err_output", "err_input"))

        gd_relu3 = GradientUnitFactory.create(
            self.parent["relu3"], "gd_relu3")
        gd_relu3.link_from(gd_pool3)
        gd_relu3.link_attrs(gd_pool3, ("err_output", "err_input"))

        gd_conv3 = GradientUnitFactory.create(
            self.parent["conv3"], "gd_conv3",
            learning_rate=0.001, learning_rate_bias=0.001,
            weights_decay=0.004, weights_decay_bias=0.004,
            gradient_moment=0.9, gradient_moment_bias=0.9)
        gd_conv3.link_from(gd_relu3)
        gd_conv3.link_attrs(gd_relu3, ("err_output", "err_input"))

        # BACK LAYER 2: CONV
        gd_norm2 = GradientUnitFactory.create(
            self.parent["norm2"], name="gd_norm2")
        gd_norm2.link_from(gd_conv3)
        gd_norm2.link_attrs(gd_conv3, ("err_output", "err_input"))

        gd_pool2 = GradientUnitFactory.create(
            self.parent["pool2"], "gd_pool2")
        gd_pool2.link_from(gd_norm2)
        gd_pool2.link_attrs(gd_norm2, ("err_output", "err_input"))

        gd_relu2 = GradientUnitFactory.create(
            self.parent["relu2"], "gd_relu2")
        gd_relu2.link_from(gd_pool2)
        gd_relu2.link_attrs(gd_pool2, ("err_output", "err_input"))

        gd_conv2 = GradientUnitFactory.create(
            self.parent["conv2"], "gd_conv2",
            learning_rate=0.001, learning_rate_bias=0.002,
            weights_decay=0.004, weights_decay_bias=0.004,
            gradient_moment=0.9, gradient_moment_bias=0.9)
        gd_conv2.link_from(gd_relu2)
        gd_conv2.link_attrs(gd_relu2, ("err_output", "err_input"))

        # BACK LAYER 1: CONV
        gd_norm1 = GradientUnitFactory.create(
            self.parent["norm1"], "gd_norm1")
        gd_norm1.link_from(gd_conv2)
        gd_norm1.link_attrs(gd_conv2, ("err_output", "err_input"))

        gd_relu1 = GradientUnitFactory.create(
            self.parent["relu1"], "gd_relu1")
        gd_relu1.link_from(gd_norm1)
        gd_relu1.link_attrs(gd_norm1, ("err_output", "err_input"))

        gd_pool1 = GradientUnitFactory.create(
            self.parent["pool1"], "gd_pool1")
        gd_pool1.link_from(gd_relu1)
        gd_pool1.link_attrs(gd_relu1, ("err_output", "err_input"))

        gd_conv1 = GradientUnitFactory.create(
            self.parent["conv1"], "gd_conv1",
            learning_rate=0.001, learning_rate_bias=0.002,
            weights_decay=0.004, weights_decay_bias=0.004,
            gradient_moment=0.9, gradient_moment_bias=0.9)
        gd_conv1.link_from(gd_pool1)
        gd_conv1.link_attrs(gd_pool1, ("err_output", "err_input"))

    def _fill_weights(self, iteration):
        for name in ["conv1", "conv2", "conv3"]:
            weights = self._load_blob(name, PropType.forward,
                                      WhenTaken.before, "blob_0", iteration)
            biases = self._load_blob(name, PropType.forward,
                                     WhenTaken.before, "blob_1", iteration)
            conv_elm = self.parent[name]
            conv_elm.weights.map_write()
            conv_elm.bias.map_write()
            conv_elm.weights.mem[:] = weights.reshape(
                conv_elm.weights.mem.shape)[:]
            conv_elm.bias.mem[:] = biases.ravel()[:]

        ip_sm_weights = self._load_blob("ip1", PropType.forward,
                                        WhenTaken.before, "blob_0", iteration)
        ip_sm_weights = ip_sm_weights.reshape(
            10, 64, 4, 4).swapaxes(1, 2).swapaxes(2, 3)
        ip_sm_biases = self._load_blob("ip1", PropType.forward,
                                       WhenTaken.before, "blob_1", iteration)

        ip_sm = self.parent["ip1"]
        ip_sm.weights.map_write()
        ip_sm.bias.map_write()
        ip_sm.weights.mem[:] = ip_sm_weights.reshape(10, 1024)[:]
        ip_sm.bias.mem[:] = ip_sm_biases.reshape(10)[:]

    def _print_deltas(self, _iter, raise_errors=False):
        """
        Prints the difference between our results and the CAFFE ones.
        Raises an error, if they are more, than 1%.
        """
        max_delta = 1.

        self.info(">>> iter %i" % _iter)
        names = ["conv1", "pool1", "relu1", "norm1", "conv2", "relu2", "pool2",
                 "norm2", "conv3"]
        for name in names:
            elm = self.parent[name]
            conv_top = self._load_blob(name, PropType.forward,
                                       WhenTaken.after, "top_0", _iter)
            elm.output.map_read()
            self.info(">> %s top delta: %.12f%%" %
                      (name, self._diff(elm.output.mem, conv_top)))

        ip_sm = self.parent["ip1"]
        ip_sm.output.map_read()

        ip_sm_top = self._load_blob(
            "loss", PropType.forward, WhenTaken.after, "top_0", _iter).reshape(
            self.n_pics, self.n_classes)

        self.info(">> ip1 top delta: %.12f%%" %
                  self._diff(ip_sm.output.mem, ip_sm_top))

        gd_ip_sm = self.parent["gd_ip1"]
        gd_ip_sm.err_input.map_read()
        gd_ip_sm_bot_err = self._load_blob(
            "ip1", WhenTaken.after, PropType.backward, "bottom_err_0", _iter)
        self.info(">> gd_ip1 bot_err delta: %.12f%%" %
                  self._diff(gd_ip_sm.err_input.mem, gd_ip_sm_bot_err))

        for name in reversed(names):
            gd_name = "gd_" + name
            gd_elm = self.parent[gd_name]
            gd_bot_err = self._load_blob(
                name, PropType.backward, WhenTaken.after,
                "bottom_err_0", _iter)
            gd_elm.err_input.map_read()
            self.info(">> %s bot_err delta: %.12f%%" %
                      (gd_name, self._diff(gd_elm.err_input.mem, gd_bot_err)))

        # Items with weights:
        conv_names = ["conv1", "conv2", "conv3"]
        for name in conv_names:
            conv_elm = self.parent[name]

            conv_weights = self._load_blob(
                name, PropType.forward, WhenTaken.before, "blob_0",
                _iter + 1).reshape(conv_elm.weights.mem.shape)
            conv_biases = self._load_blob(
                name, PropType.forward, WhenTaken.before, "blob_1",
                _iter + 1).ravel()

            conv_elm.weights.map_read()
            conv_elm.bias.map_read()
            weight_delta = self._diff(conv_elm.weights.mem, conv_weights)

            self.assertLess(weight_delta, max_delta,
                            "Result differs by %.6f" % (weight_delta))

            self.info(">>> %s weights delta: %.12f%%" % (name, weight_delta))

            bias_delta = self._diff(conv_elm.bias.mem, conv_biases)
            self.assertLess(bias_delta, max_delta,
                            "Result differs by %.6f" % (bias_delta))

            self.info(">>> %s biases delta: %.12f%%" % (name, bias_delta))

        ip_sm = self.parent["ip1"]
        ip_sm_weights = self._load_blob("ip1", PropType.forward,
                                        WhenTaken.before, "blob_0", _iter + 1)

        ip_sm_weights = ip_sm_weights.reshape(
            10, 64, 4, 4).swapaxes(1, 2).swapaxes(2, 3).reshape(10, 1024)

        ip_sm.weights.map_read()
        weight_delta = self._diff(ip_sm.weights.mem, ip_sm_weights)
        self.assertLess(weight_delta, max_delta,
                        "Result differs by %.6f" % (weight_delta))
        self.info(">>> ip1 weights delta %.12f%%" % weight_delta)

        ip_sm.bias.map_read()
        ip_sm_biases = self._load_blob(
            "ip1", PropType.forward, WhenTaken.before,
            "blob_1", _iter + 1).ravel()
        bias_delta = self._diff(ip_sm.bias.mem, ip_sm_biases)
        self.assertLess(
            bias_delta, max_delta, "Result differs by %.6f" % bias_delta)
        self.info(">>> ip1 biases delta %.12f%%" % bias_delta)

    def test_all(self):
        """
        Test forward and backward propagations of CIFAR model
        """
        self.n_pics = 3
        self.n_classes = 10
        cur_iter = 0
#        self._compress_snapshots()
        self._prepare_snapshots()

        self._create_fwd_units(cur_iter)
        self._create_gd_units(cur_iter)
        self.parent.end_point.link_from(self.parent["gd_conv1"])

        self.parent.initialize(device=self.device)
        self.info("self.parent.initialize() completed")

        self._load_labels_and_data(cur_iter)
        self.info("_load_labels_and_data() completed")

        self._fill_weights(cur_iter)
        self.info("_fill_weights() completed")

        self._load_labels_and_data(cur_iter)
        self.info("_load_labels_and_data() completed")

        self.info("Will execute run() now")
        self.parent.run()
        self.assertIsNone(self.parent.thread_pool.failure)
        self.info("run() completed")
        self._print_deltas(cur_iter)
        self.parent.stopped = False

        cur_iter = 1
        self._load_labels_and_data(cur_iter)
        self.parent.run()
        self.assertIsNone(self.parent.thread_pool.failure)
        self._print_deltas(cur_iter)
        self.parent.stopped = False

        cur_iter = 2
        self._load_labels_and_data(cur_iter)
        self.parent.run()
        self.assertIsNone(self.parent.thread_pool.failure)
        self._print_deltas(cur_iter)
        self.parent.stopped = False


if __name__ == "__main__":
    StandardTest.main()
