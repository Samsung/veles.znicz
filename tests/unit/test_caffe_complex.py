"""
Created on June 06, 2014

A complex test: comparison with CAFFE on full CIFAR model

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from enum import IntEnum
import logging
import numpy as np
import unittest
import os
import tarfile

from veles.znicz.tests.unit import standard_test
from veles.formats import Vector

from veles.znicz import activation
from veles.znicz import all2all, conv, pooling, normalization

from veles.znicz import evaluator
from veles.znicz import gd


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


class ComplexTest(standard_test.StandardTest):

    def __init__(self, methodName='runTest'):
        self.layer_dict = {}
        super(ComplexTest, self).__init__(methodName)

    def _compress_snapshots(self):
        out_archive = tarfile.open(name=os.path.join(
            self.data_dir_path, "cifar_export.xz"), mode='w|xz')
        out_archive.add(name=os.path.join(
            self.data_dir_path, "cifar_export"), arcname="cifar_export")
        out_archive.close()

    def _extract_snapshots(self):
        in_archive = tarfile.open(name=os.path.join(
            self.data_dir_path, "cifar_export.xz"), mode="r|xz")
        in_archive.extractall(self.data_dir_path)

    def _prepare_snapshots(self):
        """
        Reads a CIFAR export directory. Fills layer info path dictionary.
        """
#        self._extract_snapshots()

        in_dir = os.path.join(self.data_dir_path, "cifar_export/0")
        filenames = os.listdir(in_dir)
        layer_infos = []
        for name in filenames:
            nibbles = name.split(".")

            info = LayerInfo(nibbles[1], nibbles[0], PropType[nibbles[2]],
                             WhenTaken[nibbles[3]], time=int(nibbles[4]))
            info.path = os.path.join(in_dir, name)

            layer_infos.append(info)
        layer_infos.sort(key=lambda x: x.time)

        for info in layer_infos:
            if info.name not in self.layer_dict:
                self.layer_dict[info.name] = {}
            this_layer_dict = self.layer_dict[info.name]
            if info.propagation not in this_layer_dict:
                this_layer_dict[info.propagation] = {}
            this_prop_dict = this_layer_dict[info.propagation]
            this_prop_dict[info.when_taken] = info.path

    def _load_blob(self, layer_name, prop, when_taken, blob_name):
        """
        Loads blob by its name and parameters.

        Args:
            layer_name(str): layer name in CAFFE
            prop(:class:`PropType`) type of propagation: fwd or backward
            when_taken(:class:`WhenTaken`): was snapshot taken before or after
            blob_name(std): blob name in snapshot file

        Returns:
            :class:`numpy.ndarray(dtype=float64)`
        """
        snapshot_path = self._snapshot_path(layer_name, prop, when_taken)
        lines = self._read_lines_by_abspath(snapshot_path)
        return self._read_array(blob_name, lines)

    def _snapshot_path(self, layer_name, prop_type, when_taken):
        return self.layer_dict[layer_name][prop_type][when_taken]

    def _odds(self, array1, array2):
        """
        Calculates a difference between two ndarrays

        Args:
            array1(:class:`numpy.ndarray`):
            array2(:class:`numpy.ndarray`):

        Returns:
            float
        """
        return np.sum(np.abs(array1 - array2)) / np.sum(np.abs(array1))

    def _test_forward(self):
        """
        Test forward propagation of CIFAR model
        """

        self._prepare_snapshots()

        #FWD PROP
        #LAYER 1: CONV + MAX POOLING
        conv1 = conv.Conv(self.workflow, kx=5, ky=5, padding=(2, 2, 2, 2),
                          sliding=(1, 1), n_kernels=32,
                          weights_filling="gaussian", weights_stddev=10 ** -4,
                          bias_filling="constant", bias_stddev=0)
        conv1_weights = self._load_blob("conv1", PropType.forward,
                                        WhenTaken.before, "blob_0")
        conv1_biases = self._load_blob("conv1", PropType.forward,
                                       WhenTaken.before, "blob_1")

        conv1_bottom = self._load_blob("conv1", PropType.forward,
                                       WhenTaken.before, "bottom_0")
        conv1_top = self._load_blob("conv1", PropType.forward,
                                    WhenTaken.after, "top_0")
        conv1.input = Vector(conv1_bottom)
        conv1.link_from(self.workflow.start_point)

        relu1 = activation.ForwardStrictRELU(self.workflow)
        relu1.link_from(conv1)
        relu1.link_attrs(conv1, ("input", "output"))

        pool1 = pooling.MaxPooling(self.workflow, kx=3, ky=3, sliding=(2, 2))
        pool1.link_from(relu1)
        pool1.link_attrs(relu1, ("input", "output"))

        norm1 = normalization.LRNormalizerForward(
            self.workflow, n=5, alpha=5 * 10 ** -5, beta=0.75, k=1)
        norm1.link_from(pool1)
        norm1.link_attrs(pool1, ("input", "output"))

        #LAYER 2: CONV + AVG POOLING
        conv2 = conv.Conv(self.workflow, kx=5, ky=5, padding=(2, 2, 2, 2),
                          sliding=(1, 1), n_kernels=32,
                          weights_filling="gaussian", weights_stddev=10 ** -2,
                          bias_filling="constant", bias_stddev=0)

        conv2_weights = self._load_blob("conv2", PropType.forward,
                                        WhenTaken.before, "blob_0")
        conv2_biases = self._load_blob("conv2", PropType.forward,
                                       WhenTaken.before, "blob_1")

        conv2.link_from(norm1)
        conv2.link_attrs(norm1, ("input", "output"))

        relu2 = activation.ForwardStrictRELU(self.workflow)
        relu2.link_from(conv2)
        relu2.link_attrs(conv2, ("input", "output"))

        pool2 = pooling.AvgPooling(self.workflow, kx=3, ky=3, sliding=(2, 2))
        pool2.link_from(relu2)
        pool2.link_attrs(relu2, ("input", "output"))

        norm2 = normalization.LRNormalizerForward(
            self.workflow, n=5, alpha=5 * 10 ** -5, beta=0.75, k=1)
        norm2.link_from(pool2)
        norm2.link_attrs(pool2, ("input", "output"))

        #LAYER 3: CONV + AVG POOLING
        conv3 = conv.Conv(self.workflow, kx=5, ky=5, padding=(2, 2, 2, 2),
                          sliding=(1, 1), n_kernels=64,
                          weights_filling="gaussian", weights_stddev=10 ** -2,
                          bias_filling="constant", bias_stddev=0)

        conv3_weights = self._load_blob("conv3", PropType.forward,
                                        WhenTaken.before, "blob_0")
        conv3_biases = self._load_blob("conv3", PropType.forward,
                                       WhenTaken.before, "blob_1")

#        conv3_top = self._load_blob("conv3", PropType.forward,
#                                    WhenTaken.after, "top_0")
        conv3.link_from(norm2)
        conv3.link_attrs(norm2, ("input", "output"))

        relu3 = activation.ForwardStrictRELU(self.workflow)
        relu3.link_from(conv3)
        relu3.link_attrs(conv3, ("input", "output"))

        pool3 = pooling.AvgPooling(self.workflow, kx=3, ky=3, sliding=(2, 2))
        pool3.link_from(relu3)
        pool3.link_attrs(relu3, ("input", "output"))

        pool3_top = self._load_blob("pool3", PropType.forward,
                                    WhenTaken.after, "top_0")

        #LAYER 4: FULLY CONNECTED
        ip_sm = all2all.All2AllSoftmax(
            self.workflow, output_shape=10,
            weights_filling="uniform", weights_stddev=10 ** -2,
            bias_filling="constant", bias_stddev=0,
            )

        ip_sm_top = self._load_blob("loss", PropType.forward,
                                    WhenTaken.after, "top_0")

        ip_sm_weights = self._load_blob("ip1", PropType.forward,
                                        WhenTaken.before, "blob_0")
        ip_sm_weights = ip_sm_weights.reshape(
            10, 64, 4, 4).swapaxes(1, 2).swapaxes(2, 3)

        ip_sm_biases = self._load_blob("ip1", PropType.forward,
                                       WhenTaken.before, "blob_1")
        ip_sm.link_from(pool3)
        ip_sm.link_attrs(pool3, ("input", "output"))

        #########################
        self.workflow.end_point.link_from(ip_sm)
        self.workflow.initialize(device=self.device)

        #Weight initialization
        #L1 weights
        conv1.weights.map_write()
        conv1.bias.map_write()
        conv1.weights.mem[:] = conv1_weights.reshape(32, 75)[:]
        conv1.bias.mem[:] = conv1_biases.ravel()[:]

        #L2 weights
        conv2.weights.map_write()
        conv2.bias.map_write()
        conv2.weights.mem[:] = conv2_weights.reshape(
            conv2.n_kernels, conv2.kx * conv2.ky * conv1.n_kernels)[:]
        conv2.weights.bias = conv2_biases.ravel()[:]

        #L3 weights
        conv3.weights.map_write()
        conv3.bias.map_write()
        conv3.weights.mem[:] = conv3_weights.reshape(
            conv3.n_kernels, conv3.kx * conv3.ky * conv2.n_kernels)[:]
        conv3.weights.bias = conv3_biases.ravel()[:]

        #L4 weights
        ip_sm.weights.map_write()
        ip_sm.bias.map_write()
        ip_sm.weights.mem[:] = ip_sm_weights.reshape(10, 1024)[:]
        ip_sm.bias.mem[:] = ip_sm_biases.reshape(10)[:]

        ##########################
        self.workflow.end_point.link_from(ip_sm)
        self.workflow.run()
        ##########################

        #Comparing with CAFFE data

        conv1.output.map_read()
        logging.info("conv1 delta %f" %
                     self._odds(conv1_top, conv1.output.mem))

        norm1_top = self._load_blob("norm1", PropType.forward,
                                    WhenTaken.after, "top_0")
        norm1.output.map_read()
        logging.info("norm1 delta %f" %
                     self._odds(norm1_top, norm1.output.mem))

        norm2_top = self._load_blob("norm2", PropType.forward,
                                    WhenTaken.after, "top_0")
        norm2.output.map_read()
        logging.info("norm2 delta %f" %
                     self._odds(norm2_top, norm2.output.mem))

        pool3.output.map_read()
        logging.info("pool3 delta %f" %
                     self._odds(pool3_top, pool3.output.mem))

        ip_sm.output.map_read()
        logging.info("ip_sm delta %f" %
                     self._odds(ip_sm_top, ip_sm.output.mem))

    def test_backward(self):
        """
        Test back propagation of CIFAR model
        """

        n_pics = 3
        n_classes = 10

        self._prepare_snapshots()

        # FWD LAYER 4: FULLY CONNECTED + SOFTMAX
        ip_sm = all2all.All2AllSoftmax(
            self.workflow, output_shape=10,
            weights_filling="uniform", weights_stddev=10 ** -2,
            bias_filling="constant", bias_stddev=0,
            )
        ip_sm_top = self._load_blob("loss", PropType.forward,
                                    WhenTaken.after, "top_0")
        ip_sm_bottom = self._load_blob("ip1", PropType.forward,
                                       WhenTaken.before, "bottom_0")
        ip_sm.input = Vector(ip_sm_bottom)
        ip_sm_weights = self._load_blob("ip1", PropType.forward,
                                        WhenTaken.before, "blob_0")
        ip_sm_weights = ip_sm_weights.reshape(
            10, 64, 4, 4).swapaxes(1, 2).swapaxes(2, 3)
        ip_sm_biases = self._load_blob("ip1", PropType.forward,
                                       WhenTaken.before, "blob_1")
        ip_sm.link_from(self.workflow.start_point)

        #BACK LAYER 4: EV + GD_A2ASM
        labels = self._load_blob(
            "loss", PropType.backward, WhenTaken.before,
            "bottom_1").astype(np.int32).ravel()

        ev = evaluator.EvaluatorSoftmax(self.workflow)
        ev.link_from(ip_sm)
        ev.batch_size = n_pics
        ev.link_attrs(ip_sm, "max_idx")
        ev.labels = Vector(labels)
        ev.link_attrs(ip_sm, "output")

        ev_bot_err = self._load_blob(
            "loss", PropType.backward, WhenTaken.after,
            "bottom_err_0").reshape(n_pics, n_classes) * n_pics

        gd_ip_sm = gd.GDSM(self.workflow)
        gd_ip_sm.input = Vector(self._load_blob("ip1", PropType.forward,
                                                WhenTaken.before, "bottom_0"))
        gd_ip_sm.output = Vector(self._load_blob("ip1", PropType.forward,
                                                 WhenTaken.after, "top_0"))
        gd_ip_sm.batch_size = n_pics
        gd_ip_sm.link_from(ev)
        gd_ip_sm.link_attrs(ev, "err_output")
        gd_ip_sm.link_attrs(ip_sm, "weights")
        gd_ip_sm.link_attrs(ip_sm, "bias")

        gd_ip_sm_bot_err = self._load_blob("ip1", PropType.backward,
                                           WhenTaken.after, "bottom_err_0")

        ####################
        #INITIALIZATION
        self.workflow.end_point.link_from(ev)
        self.workflow.initialize(device=self.device)
        ####################
        #SETTING WEIGHTS
        ip_sm.weights.map_write()
        ip_sm.bias.map_write()
        ip_sm.weights.mem[:] = ip_sm_weights.reshape(10, 1024)[:]
        ip_sm.bias.mem[:] = ip_sm_biases.reshape(10)[:]

        ######################
        self.workflow.run()
        ######################

        ip_sm.output.map_read()
#        print(ip_sm.output.mem)
#        print(ip_sm_top)
        logging.info("ip_sm delta %f" %
                     self._odds(ip_sm.output.mem, ip_sm_top))

        ev.err_output.map_read()
        logging.info("sm_err delta %f" %
                     self._odds(ev.err_output.mem, ev_bot_err))

        gd_ip_sm.err_input.map_read()
        logging.info("gd_ip_sm delta %f" %
                     self._odds(gd_ip_sm.err_input.mem, gd_ip_sm_bot_err))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("COMPLEX TEST")
    unittest.main()
