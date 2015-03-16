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
    MAPPING = "cifar_loader"

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

    def create_workflow(self):
        self.link_repeater(self.start_point)

        self.link_loader(self.repeater)

        self.link_forwards(("input", "minibatch_data"), self.loader)

        self.link_evaluator(self.forwards[-1])

        self.link_decision(self.evaluator)

        end_units = [self.link_snapshotter(self.decision)]

        if root.cifar.image_saver.do:
            end_units.append(self.link_image_saver(self.decision))

        if root.cifar.add_plotters:
            end_units.extend(link(self.decision) for link in (
                self.link_error_plotter, self.link_conf_matrix_plotter,
                self.link_err_y_plotter))
            end_units.append(self.link_weights_plotter(
                root.cifar.layers, root.cifar.weights_plotter.limit,
                "weights", self.decision))

            self.link_gds(*end_units)
            last = self.link_table_plotter(root.cifar.layers, self.gds[0])
        else:
            last = self.link_gds(*end_units)

        if root.cifar.learning_rate_adjust.do:
            last = self.link_lr_adjuster(last)
        self.repeater.link_from(last)

        self.link_end_point(*end_units)


def run(load, main):
    load(CifarWorkflow,
         decision_config=root.cifar.decision,
         snapshotter_config=root.cifar.snapshotter,
         image_saver_config=root.cifar.image_saver,
         loader_config=root.cifar.loader,
         similar_weights_plotter_config=root.cifar.similar_weights_plotter,
         layers=root.cifar.layers,
         loader_name=root.cifar.loader_name,
         loss_function=root.cifar.loss_function)
    main()
