#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on Apr 18, 2014

@author: Alexey Golovizin <a.golovizin@samsung.com>
"""

from veles.config import root
from veles.znicz import nn_units

class Workflow(nn_units.NNWorkflow):
    def __init__(self):
        pass

def run(load, main):
    load(Workflow, layers=root.imagenet.layers)
    main(global_alpha=root.imagenet.global_alpha,
         global_lambda=root.imagenet.global_lambda,
         minibatch_maxsize=root.loader.minibatch_maxsize)