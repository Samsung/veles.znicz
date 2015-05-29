# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jul 3, 2013

Model created for object recognition. Dataset - CIFAR10. Self-constructing
Model. It means that Model can change for any Model (Convolutional, Fully
connected, different parameters) in configuration file.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
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
        self.link_downloader(self.start_point)
        self.link_repeater(self.downloader)
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
                "weights", self.decision))

            self.link_gds(*end_units)
            last = self.gds[0]
        else:
            last = self.link_gds(*end_units)
        if root.cifar.lr_adjuster.do:
            last = self.link_lr_adjuster(last)
        self.repeater.link_from(last)

        self.link_end_point(last)


def run(load, main):
    load(CifarWorkflow,
         decision_config=root.cifar.decision,
         snapshotter_config=root.cifar.snapshotter,
         image_saver_config=root.cifar.image_saver,
         loader_config=root.cifar.loader,
         similar_weights_plotter_config=root.cifar.similar_weights_plotter,
         layers=root.cifar.layers,
         downloader_config=root.cifar.downloader,
         loader_name=root.cifar.loader_name,
         loss_function=root.cifar.loss_function,
         weights_plotter_config=root.cifar.weights_plotter,
         lr_adjuster_config=root.cifar.lr_adjuster)
    main()
