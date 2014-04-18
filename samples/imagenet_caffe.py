#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on Apr 18, 2014

@author: Alexey Golovizin <a.golovizin@samsung.com>
"""

from veles.config import root, get
from veles.znicz import nn_units
from veles.znicz import loader
from veles.znicz import conv
from veles.znicz import pooling
from veles.znicz import all2all

class Loader(loader.FullBatchLoader):
    pass

class Workflow(nn_units.NNWorkflow):
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.rpt.link_from(self.start_point)

        self.loader = Loader(self)
        self.loader.link_from(self.rpt)

        # LAYER 1
        conv1 = conv.ConvRELU(self, n_kernels=96, kx=11, ky=11,
            sliding=(4, 4, 4, 4), padding=(0, 0, 0, 0), device=device)
        conv1.link_from(self.loader)
        conv1.link_attrs(self.loader, ("input", "minibatch_data"))
        self.forward.append(conv1)

        pool1 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
            device=device)
        pool1.link_from(conv1)
        pool1.link_attrs(conv1, ("input", "output"))
        self.forward.append(pool1)

        # TODO: normalization, gaussian filling

        # LAYER 2
        conv2 = conv.ConvRELU(self, n_kernels=256, kx=5, ky=5,
            sliding=(1, 1, 1, 1), padding=(2, 2, 2, 2), device=device)
        conv2.link_from(pool1)
        conv2.link_attrs(pool1, ("input", "output"))
        self.forward.append(conv2)

        pool2 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
            device=device)
        pool1.link_from(conv2)
        pool1.link_attrs(conv2, ("input", "output"))
        self.forward.append(pool2)

        # LAYER 3
        conv3 = conv.ConvRELU(self, n_kernels=384, kx=3, ky=3,
            sliding=(1, 1, 1, 1), padding=(1, 1, 1, 1), device=device)
        conv3.link_from(pool2)
        conv3.link_attrs(pool2, ("input", "output"))
        self.forward.append(conv3)

        # LAYER 4
        conv4 = conv.ConvRELU(self, n_kernels=384, kx=3, ky=3,
            sliding=(1, 1, 1, 1), padding=(1, 1, 1, 1), device=device)
        conv3.link_from(conv3)
        conv3.link_attrs(conv3 ("input", "output"))
        self.forward.append(conv4)

        # LAYER 5
        conv5 = conv.ConvRELU(self, n_kernels=256, kx=3, ky=3,
            sliding=(1, 1, 1, 1), padding=(1, 1, 1, 1), device=device)
        conv3.link_from(conv4)
        conv3.link_attrs(conv4 ("input", "output"))
        self.forward.append(conv5)

        pool5 = pooling.MaxPooling(self, kx=3, ky=3, sliding=(2, 2),
            device=device)
        pool1.link_from(conv5)
        pool1.link_attrs(self.conv5, ("input", "output"))
        self.forward.append(pool5)

        # LAYER 6
        fc6 = all2all.All2AllRELU(self, output_shape=4096)
        fc6.link_from(pool5)
        fc6.link_attrs(pool5, ("input", "output"))
        self.forward.append(fc6)

            # TODO: dropout

        # LAYER 7
        fc7 = all2all.All2AllRELU(self, output_shape=4096)
        fc7.link_from(fc6)
        fc7.link_attrs(self.fc6, ("input", "output"))
        self.forward.append(fc7)

            # TODO: dropout

        # LAYER 8
        fc8 = all2all.All2AllSoftmax(self, output_shape=4096)
        fc8.link_from(fc7)
        fc8.link_attrs(self.fc7, ("input", "output"))
        self.forward.append(fc8)


    def initialize(self, global_alpha, global_lambda, minibatch_maxsize, device):
        return super(Workflow, self).initialize()

def run(load, main):
    load(Workflow, layers=root.imagenet.layers)
    main(global_alpha=get(root.imagenet.global_alpha),
         global_lambda=root.imagenet.global_lambda,
         minibatch_maxsize=root.loader.minibatch_maxsize)
