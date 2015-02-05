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
from veles.loader import PicklesImageFullBatchLoader
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
        self.link_repeater(self.start_point)

        self.link_loader(self.repeater)

        self.link_forwards(self.loader, ("input", "minibatch_data"))

        self.link_evaluator(self.forwards[-1])

        self.link_decision(self.evaluator)

        self.link_snapshotter(self.decision)

        if root.cifar.image_saver.do:
            self.link_image_saver(self.snapshotter)

            self.link_gds(self.image_saver)
        else:
            self.link_gds(self.snapshotter)

        if root.cifar.add_plotters:
            self.link_error_plotter(self.gds[0])

            self.link_conf_matrix_plotter(self.error_plotter[-1])

            self.link_err_y_plotter(self.conf_matrix_plotter[-1])

            self.link_weights_plotter(
                self.err_y_plotter[-1], layers=root.cifar.layers,
                limit=root.cifar.weights_plotter.limit,
                weights_input="weights")

            self.link_table_plotter(
                root.cifar.layers, self.weights_plotter[-1])

            last = self.table_plotter
        else:
            last = self.gds[0]

        if root.cifar.learning_rate_adjust.do:
            self.link_lr_adjuster(last)

            self.link_end_point(self.lr_adjuster)
        else:
            self.link_end_point(last)


def run(load, main):
    load(CifarWorkflow,
         decision_config=root.cifar.decision,
         snapshotter_config=root.cifar.snapshotter,
         image_saver_config=root.cifar.image_saver,
         similar_weights_plotter_config=root.cifar.similar_weights_plotter,
         layers=root.cifar.layers,
         loss_function=root.cifar.loss_function)
    main()
