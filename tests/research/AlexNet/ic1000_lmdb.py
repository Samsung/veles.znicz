#!/usr/bin/python3 -O
"""
Created on Dec 19, 2014

Copyright (c) 2014 Samsung R&D Institute Russia
"""


from veles.config import root
from veles.znicz.standard_workflow import StandardWorkflow

from veles.znicz.loader import loader_lmdb  # pylint: disable=W0611


class ImagenetLMDBWorkflow(StandardWorkflow):
    """
    Imagenet Workflow
    """

    def create_workflow(self):
        # Add repeater unit
        self.link_repeater(self.start_point)

        # Add loader unit
        self.link_loader(self.repeater)

        # Add fwds units
        self.link_forwards(self.loader, ("input", "minibatch_data"))

        # Add evaluator for single minibatch
        self.link_evaluator(self.forwards[-1])

        # Add decision unit
        self.link_decision(self.evaluator)

        # Add snapshotter unit
        self.link_snapshotter(self.decision)

        # Add gradient descent units
        self.link_gds(self.snapshotter)

        if root.imagenet.add_plotters:
            # Add error plotter unit
            self.link_error_plotter(self.gds[0])

            # Add Confusion matrix plotter unit
            self.link_conf_matrix_plotter(self.error_plotter[-1])

            # Add Err y plotter unit
            self.link_err_y_plotter(self.conf_matrix_plotter[-1])

            # Add Weights plotter unit
            self.link_weights_plotter(
                self.err_y_plotter[-1], layers=root.imagenet.layers,
                limit=root.imagenet.weights_plotter.limit,
                weights_input="weights")

            last = self.weights_plotter[-1]
        else:
            last = self.gds[0]

        # Add end_point unit
        self.link_end_point(last)


def run(load, main):
    load(ImagenetLMDBWorkflow,
         loader_name=root.imagenet.loader_name,
         loader_config=root.imagenet.loader,
         decision_config=root.imagenet.decision,
         snapshotter_config=root.imagenet.snapshotter,
         layers=root.imagenet.layers,
         loss_function=root.imagenet.loss_function)
    main()
