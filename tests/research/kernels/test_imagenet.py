"""
Created on Aug 13, 2014

Data archiving.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from veles.formats import Vector
import os
import unittest
from veles.config import root
from veles import opencl
from veles.tests import DummyLauncher
from veles.znicz.tests.research.kernels.imaganet_caffe_forward import\
    Workflow as ImagenetWorkflow
from veles.znicz.tests.research.kernels.tar_unpacker import extract
from veles.znicz.tests.test_utils import read_caffe_array,\
    read_lines_by_abspath


class TestCaffe(unittest.TestCase):
    def setUp(self):
        self.data_path = os.path.dirname(os.path.realpath(__file__))
        extract("/data/veles/datasets/imagenet/dumps", self.data_path)
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.bottom_path = os.path.join(self.data_path, "data/bottom_caffe")
        self.top_path = os.path.join(self.data_path, "data/top_caffe")
        self.weight_path = os.path.join(self.data_path, "data/weights_caffe")
        self.layers_names = ["conv1", "relu1", "pool1", "norm1", "conv2",
                             "relu2", "pool2", "norm2", "conv3", "relu3",
                             "conv4", "relu4", "conv5", "relu5", "pool5",
                             "fc6", "relu6", "fc7", "relu7", "fcsm8"]
        self.device = opencl.Device()
        self.top_workflow = DummyLauncher()

    @staticmethod
    def mean_diff(array1, array2):
        mean = np.abs(array1) + np.abs(array2)
        mean = np.mean(mean) / 2
        return np.mean(np.abs(array1 - array2)) / mean

    @staticmethod
    def image_drawing(data, name_f=''):
        n = int(np.ceil(np.sqrt(data.shape[3])))
        fig1 = plt.figure()
        fig1.suptitle(name_f)
        for i in range(data.shape[3]):
            ax1 = fig1.add_subplot(n, n, i + 1)
            ax1.imshow(data[0, :, ::-1, i], interpolation='none', cmap="gray")
            ax1.get_xaxis().set_ticks([])
            ax1.get_yaxis().set_ticks([])

    def generate_workflow(self, weight_path, type_loader="veles"):
        """
        Generate workflow for imagenet caffe network, which classified image
        This function doesn't work if cheking, that train data is'n empty
        is  enabled
        Args:
            weight_path(str): file this weights from caffe imagenet, which were
            obtained with help AppendToFile function
            type_loader(str): string with two values: veles and caffe.
                This variable indicates who forms the entrance to conv1

        Returns:
            wfl(ImagenetWorkflow): ready for classification network
        """

        wfl = ImagenetWorkflow(self.top_workflow, layers=None,
                               device=self.device)
#         unlink loader and loader and load  data for dedug may be used
        to_fail = True
        if type_loader == "caffe":
            wfl.loader.unlink_all()
            lines = read_lines_by_abspath(
                os.path.join(self.bottom_path, "conv1"))
            img = read_caffe_array("blob_0", lines)
            wfl["conv1"].input = Vector(img[:, :, :, :])
            wfl["conv1"].link_from(wfl.start_point)
            to_fail = False
        elif type_loader == "veles":
            to_fail = False
        self.assertEqual(
            to_fail, False,
            "type_loader %s doesn't equal veles or caffe" % type_loader)
        wfl.generate_graph("/home/timmy/graph_img.png")
        wfl.initialize(device=self.device)
        #  initialization weights for conv layers from files
        conv_names = ("conv1", "conv2", "conv3", "conv4", "conv5")
        for name in conv_names:
            logging.info("set weights for %s" % name)
            lines = read_lines_by_abspath(os.path.join(weight_path, name))
            weights = read_caffe_array("blob_0", lines)
            bias = read_caffe_array("blob_1", lines)
            if(name == "conv2" or name == "conv4" or name == "conv5"):
                newShape = (weights.shape[0], weights.shape[1],
                            weights.shape[2], weights.shape[3] * 2)
                newWeights = np.zeros(shape=newShape)
                for i in range(newShape[0]):
                    if (i < round(weights.shape[0] / 2)):
                        newWeights[i, :, :, 0: weights.shape[3]][:] = \
                            weights[i, :, :, :][:]
                    else:
                        newWeights[i, :, :, weights.shape[3]:][:] = \
                            weights[i, :, :, :][:]
                weights = newWeights
            wfl[name].weights.map_invalidate()
            wfl[name].weights.mem[:] = \
                weights.reshape(wfl[name].weights.mem.shape)[:]
            wfl[name].bias.map_invalidate()
            wfl[name].bias.mem[:] = bias.reshape(wfl[name].bias.mem.shape)[:]
            logging.info("weights for %s have been set successfully" % name)
        # this parameters are needed for first full connection
        # layer, which is connected to the last convolutional layer
        n_neurons = {"fc6": 4096}
        # n_size: = height (suggesting that width equals height)
        n_size = {"fc6": 6}
        n_chans = {"fc6": 256}
        full_connect_names = ("fc6", "fc7", "fcsm8")
        for name in full_connect_names:
            logging.info("filling weights for %s" % name)
            lines = read_lines_by_abspath(os.path.join(weight_path, name))
            weights = []
            weights = read_caffe_array("blob_0", lines)
            if name == "fc6":
                weights = weights.transpose((1, 0, 2, 3))
                weights = weights.reshape(
                    n_neurons[name], n_chans[name], n_size[name],
                    n_size[name]).swapaxes(1, 2).swapaxes(2, 3)
                weights = weights.reshape((n_neurons[name], n_size[name] *
                                           n_size[name] * n_chans[name]))
            else:
                weights = weights[0, :, :, 0]
            bias = read_caffe_array("blob_1", lines)
            wfl[name].weights.map_invalidate()
            wfl[name].weights.mem[:] = weights[:]
            wfl[name].bias.map_invalidate()
            wfl[name].bias.mem[:] = bias.reshape((np.max(bias.shape),))[:]
            logging.info("weights for %s have been set successfully" % name)
        return wfl

    def test_caffe_loader(self):
        """
        Test full veles workflow with caffe loader.
        """
        logging.info("TEST CAFFE LOADER")
        test_eps = 1e-5
        wfl = self.generate_workflow(self.weight_path, "caffe")
        wfl.run()
        layer_number = 0
        for name in self.layers_names:
            if name == "fcsm8":
                lines_top = read_lines_by_abspath(
                    os.path.join(self.top_path, "prob"))
                lines_bottom = read_lines_by_abspath(
                    os.path.join(self.bottom_path, "fc8"))
            else:
                lines_top = read_lines_by_abspath(
                    os.path.join(self.top_path, name))
                lines_bottom = read_lines_by_abspath(
                    os.path.join(self.bottom_path, name))
            caffe_top = read_caffe_array("blob_0", lines_top)
            caffe_bottom = read_caffe_array("blob_0", lines_bottom)
            if layer_number == 15:
                caffe_top = caffe_top[:, 0, 0, :]
            elif layer_number > 15:
                caffe_bottom = caffe_bottom[:, 0, 0, :]
                caffe_top = caffe_top[:, 0, 0, :]
            wfl[name].output.map_read()
            veles_top = wfl[name].output.mem
            wfl[name].input.map_read()
            veles_bottom = wfl[name].input.mem
            top_mean_diff = TestCaffe.mean_diff(veles_top, caffe_top)
            bottom_mean_diff = TestCaffe.mean_diff(veles_bottom, caffe_bottom)
            self.assertLess(top_mean_diff, test_eps,
                            "Result differs for %s top  by %.6e" %
                            (name, top_mean_diff))
            self.assertLess(bottom_mean_diff, test_eps,
                            "Result differs for %s bottom  by %.6e" %
                            (name, bottom_mean_diff))
            logging.info("Result differs for %s TOP  by %.5e" %
                         (name, top_mean_diff))
            logging.info("Result differs for %s BOTTOM  by %.5e" %
                         (name, bottom_mean_diff))

            layer_number += 1
        del wfl

    def test_veles_loader(self):
        """
        Test full veles workflow with veles loader.
        """
        logging.info("TEST VELES LOADER")
        wfl = self.generate_workflow(self.weight_path, "veles")
        wfl.run()
        layer_number = 0
        for name in self.layers_names:
            if name == "fcsm8":
                lines_top = read_lines_by_abspath(
                    os.path.join(self.top_path, "prob"))
                lines_bottom = read_lines_by_abspath(
                    os.path.join(self.bottom_path, "fc8"))
            else:
                lines_top = read_lines_by_abspath(
                    os.path.join(self.top_path, name))
                lines_bottom = read_lines_by_abspath(
                    os.path.join(self.bottom_path, name))
            caffe_top = read_caffe_array("blob_0", lines_top)
            caffe_bottom = read_caffe_array("blob_0", lines_bottom)
            if layer_number == 15:
                caffe_top = caffe_top[:, 0, 0, :]
            elif layer_number > 15:
                caffe_bottom = caffe_bottom[:, 0, 0, :]
                caffe_top = caffe_top[:, 0, 0, :]
            wfl[name].output.map_read()
            veles_top = wfl[name].output.mem
            wfl[name].input.map_read()
            veles_bottom = wfl[name].input.mem
            top_mean_diff = TestCaffe.mean_diff(veles_top, caffe_top)
            bottom_mean_diff = TestCaffe.mean_diff(veles_bottom, caffe_bottom)
            logging.info("Result differs for %s TOP  by %.5e" %
                         (name, top_mean_diff))
            logging.info("Result differs for %s BOTTOM  by %.5e" %
                         (name, bottom_mean_diff))
        del wfl

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    suite = unittest.makeSuite(TestCaffe)
    unittest.TextTestRunner(verbosity=2).run(suite)
