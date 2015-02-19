#!/usr/bin/python3 -O
"""
Created on Dec 19, 2014

Model created for object recognition. Database - Imagenet (1000 classes).
Self-constructing Model. It means that Model can change for any Model
(Convolutional, Fully connected, different parameters) in configuration file.

Copyright (c) 2014 Samsung R&D Institute Russia
"""


from veles.config import root
from veles.znicz.standard_workflow import StandardWorkflow

from veles.znicz.loader import loader_lmdb  # pylint: disable=W0611


class ImagenetWorkflow(StandardWorkflow):
    """
    Imagenet Workflow
    """

    def create_workflow(self):
        self.link_repeater(self.start_point)
        self.link_loader(self.repeater)
        self.link_forwards(self.loader, ("input", "minibatch_data"))
        self.link_evaluator(self.forwards[-1])
        self.link_decision(self.evaluator)
        self.link_snapshotter(self.decision)
        self.link_gds(self.snapshotter)
        if root.imagenet.add_plotters:
            self.link_error_plotter(self.gds[0])
            self.link_conf_matrix_plotter(self.error_plotter[-1])
            self.link_err_y_plotter(self.conf_matrix_plotter[-1])
            self.link_weights_plotter(
                self.err_y_plotter[-1], layers=root.imagenet.layers,
                limit=root.imagenet.weights_plotter.limit,
                weights_input="weights")
            last = self.weights_plotter[-1]
        else:
            last = self.gds[0]

        self.link_end_point(last)


def run(load, main):
    load(ImagenetWorkflow,
         loader_name=root.imagenet.loader_name,
         loader_config=root.imagenet.loader,
         decision_config=root.imagenet.decision,
         snapshotter_config=root.imagenet.snapshotter,
         layers=root.imagenet.layers,
         loss_function=root.imagenet.loss_function)
    main()
