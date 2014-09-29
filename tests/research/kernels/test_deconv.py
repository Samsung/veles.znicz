"""
Create imagenet caffe network with 5 convolution and five deconvolution
layers.
Output from deconv1 is very bad.
if will be used deconvolution network with 3 layers, then result will be
satisfactory
"""
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import logging
from veles.formats import Vector
from veles import opencl
from veles.znicz.tests.test_utils import read_lines_by_abspath, \
    read_caffe_array
from veles.tests import DummyLauncher
from veles.znicz.tests.research.kernels.imagenet_deconv import Workflow as \
    ImagenetWorkflow


def image_drawing(data, name_f=''):
    n = int(np.ceil(np.sqrt(data.shape[3])))
    fig1 = plt.figure()
    fig1.suptitle(name_f)
    for j in range(data.shape[3]):
        ax1 = fig1.add_subplot(n, n, j + 1)
        ax1.imshow(data[0, :, ::-1, j], interpolation='none', cmap="gray")
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
logging.basicConfig(level=logging.DEBUG)
data_path = os.path.dirname(os.path.realpath(__file__))
weight_path = os.path.join(data_path, "data/weights_caffe/")
top_workflow = DummyLauncher()
device = opencl.Device()
wfl = ImagenetWorkflow(top_workflow, layers=None, device=device)

wfl["depool5"].link_attrs(wfl["pool5"], ("input", "output"))
wfl["depool5"].link_attrs(wfl["pool5"], ("get_output_shape_from", "input"))
wfl["depool5"].link_attrs(wfl["pool5"], ("output_offset", "input_offset"))

wfl["deconv5"].link_attrs(wfl["depool5"], ("input", "output"))
wfl["deconv5"].link_attrs(wfl["conv5"], ("get_output_shape_from", "input"))
wfl["deconv5"].link_attrs(wfl["conv5"], ("weights", "weights"))

wfl["deconv4"].link_attrs(wfl["deconv5"], ("input", "output"))
wfl["deconv4"].link_attrs(wfl["conv4"], ("get_output_shape_from", "input"))
wfl["deconv4"].link_attrs(wfl["conv4"], ("weights", "weights"))

wfl["deconv3"].link_attrs(wfl["deconv4"], ("input", "output"))
wfl["deconv3"].link_attrs(wfl["conv3"], ("get_output_shape_from", "input"))
wfl["deconv3"].link_attrs(wfl["conv3"], ("weights", "weights"))

wfl["depool2"].link_attrs(wfl["deconv3"], ("input", "output"))
wfl["depool2"].link_attrs(wfl["pool2"], ("get_output_shape_from", "input"))
wfl["depool2"].link_attrs(wfl["pool2"], ("output_offset", "input_offset"))

wfl["deconv2"].link_attrs(wfl["depool2"], ("input", "output"))
wfl["deconv2"].link_attrs(wfl["conv2"], ("get_output_shape_from", "input"))
wfl["deconv2"].link_attrs(wfl["conv2"], ("weights", "weights"))

wfl["depool1"].link_attrs(wfl["deconv2"], ("input", "output"))
wfl["depool1"].link_attrs(wfl["pool1"], ("get_output_shape_from", "input"))
wfl["depool1"].link_attrs(wfl["pool1"], ("output_offset", "input_offset"))

wfl["deconv1"].link_attrs(wfl["depool1"], ("input", "output"))
wfl["deconv1"].link_attrs(wfl["conv1"], ("get_output_shape_from", "input"))
wfl["deconv1"].link_attrs(wfl["conv1"], ("weights", "weights"))

wfl.initialize(device=device)

conv_names = ("conv1", "conv2", "conv3", "conv4", "conv5")
deconv_names = {"conv1": "deconv1", "conv2": "deconv2", "conv3": "deconv3",
                "conv4": "deconv4", "conv5": "deconv5"}
for name in conv_names:
    logging.info("initialize weights for %s" % name)
    lines = read_lines_by_abspath(os.path.join(weight_path, name))
    weights = read_caffe_array("blob_0", lines)
    bias = read_caffe_array("blob_1", lines)
    if(name == "conv2" or name == "conv4" or name == "conv5"):
        newShape = (weights.shape[0], weights.shape[1], weights.shape[2],
                    weights.shape[3] * 2)
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
    wfl[name].weights.mem[:] = weights.reshape(wfl[name].weights.mem.shape)[:]
    wfl[name].bias.map_invalidate()
    wfl[name].bias.mem[:] = bias.reshape(wfl[name].bias.mem.shape)[:]
    logging.info("weights were initialized for %s successfully" % name)
    # it is necessary to add bias in deconv
    inp = Vector(wfl[name].bias.mem[:])
    inp.initialize(wfl["deconv2"])
    wfl[deconv_names[name]].bias = inp

wfl.run()

wfl["deconv1"].output.map_read()
out_dec = wfl["deconv1"].output.mem[:]
sys.exit(0)
