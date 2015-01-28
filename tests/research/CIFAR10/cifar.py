#!/usr/bin/python3 -O
"""
Created on Jul 3, 2013

Model created for object recognition. Dataset - CIFAR10. Self-constructing
Model. It means that Model can change for any Model (Convolutional, Fully
connected, different parameters) in configuration file.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root
from veles.znicz.loader import PicklesImageFullBatchLoader
from veles.znicz.standard_workflow import StandardWorkflow


class CifarLoader(PicklesImageFullBatchLoader):
    """Loads Cifar dataset.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["color_space"] = "RGB"
        kwargs["validation_pickles"] = [root.cifar.data_paths.validation]
        kwargs["train_pickles"] = [
            os.path.join(root.cifar.data_paths.train, ("data_batch_%d" % i))
            for i in range(1, 6)]
        super(CifarLoader, self).__init__(workflow, **kwargs)

    def reshape(self, shape):
        assert shape == (3072,)
        return super(CifarLoader, self).reshape((3, 32, 32))

    def transform_data(self, data):
        return super(CifarLoader, self).transform_data(data.reshape(
            10000, 3, 32, 32))


class CifarWorkflow(StandardWorkflow):
    """
    Model created for object recognition. Dataset - CIFAR10. Self-constructing
    Model. It means that Model can change for any Model (Convolutional, Fully
    connected, different parameters) in configuration file.
    """
    def link_loader(self, init_unit):
        self.loader = CifarLoader(
            self, **root.cifar.loader.__dict__)
        self.loader.link_from(init_unit)

    def create_workflow(self):
        # Add repeater unit
        self.link_repeater(self.start_point)

        # Add loader unit
        self.link_loader(self.repeater)

        # Add fwds units
        self.link_forwards(self.loader, ("input", "minibatch_data"))

        if root.cifar.image_saver.do:
            # Add image_saver unit
            self.link_image_saver(self.forwards[-1])

            # Add evaluator unit
            self.link_evaluator(self.image_saver)
        else:
            # Add evaluator unit
            self.link_evaluator(self.forwards[-1])

        # Add decision unit
        self.link_decision(self.evaluator)

        # Add snapshotter unit
        self.link_snapshotter(self.decision)

        if root.cifar.image_saver.do:
            self.image_saver.gate_skip = ~self.decision.improved
            self.image_saver.link_attrs(self.snapshotter,
                                        ("this_save_time", "time"))

        # Add gradient descent units
        self.link_gds(self.snapshotter)

        if root.cifar.add_plotters:

            # Add error plotter unit
            self.link_error_plotter(self.gds[0])

            # Add Confusion matrix plotter unit
            self.link_conf_matrix_plotter(self.error_plotter[-1])

            # Add Err y plotter unit
            self.link_err_y_plotter(self.conf_matrix_plotter[-1])

            # Add Weights plotter unit
            self.link_weights_plotter(
                self.err_y_plotter[-1], layers=root.cifar.layers,
                limit=root.cifar.weights_plotter.limit,
                weights_input="weights")

            # Add Similar weights plotter unit
            self.link_similar_weights_plotter(
                self.weights_plotter[-1], layers=root.cifar.layers,
                limit=root.cifar.weights_plotter.limit,
                magnitude=root.cifar.similar_weights_plotter.magnitude,
                form=root.cifar.similar_weights_plotter.form,
                peak=root.cifar.similar_weights_plotter.peak)

            # Add Table plotter unit
            self.link_table_plotter(
                root.cifar.layers, self.similar_weights_plotter[-1])

            last = self.table_plotter
        else:
            last = self.gds[0]

        if root.cifar.learning_rate_adjust.do:

            # Add learning_rate_adjust unit
            self.link_lr_adjuster(last)

            # Add end_point unit
            self.link_end_point(self.lr_adjuster)
        else:
            # Add end_point unit
            self.link_end_point(last)


def run(load, main):
    load(CifarWorkflow,
         decision_config=root.cifar.decision,
         snapshotter_config=root.cifar.snapshotter,
         image_saver_config=root.cifar.image_saver,
         layers=root.cifar.layers,
         loss_function=root.cifar.loss_function)
    main()
