#!/usr/bin/python3 -O
"""
Created on Nov 14, 2014

Model created for object recognition. Database - Fake Imagenet (1000 classes
of random pictures 227*227). Self-constructing Model. It means that Model can
change for any Model (Convolutional, Fully connected, different parameters) in
configuration file.

Copyright (c) 2014 Samsung R&D Institute Russia
"""


from veles.config import root
from veles.znicz.standard_workflow import StandardWorkflow
from veles.znicz.tests.research.AlexNet.imagenet_loader import \
    ImagenetCaffeLoader


class ImagenetWorkflow(StandardWorkflow):
    """
    Model created for object recognition. Database - Fake Imagenet (1000
    classes of random pictures 227*227). Self-constructing Model. It means that
    Model can change for any Model (Convolutional, Fully connected, different
    parameters) in configuration file.
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

    def link_loader(self, init_unit):
        self.loader = ImagenetCaffeLoader(
            self, on_device=root.imagenet.loader.on_device,
            minibatch_size=root.imagenet.loader.minibatch_size,
            shuffle_limit=root.imagenet.loader.shuffle_limit,
            crop_size_sx=root.imagenet.loader.crop_size_sx,
            crop_size_sy=root.imagenet.loader.crop_size_sy,
            sx=root.imagenet.loader.sx,
            sy=root.imagenet.loader.sy,
            original_labels_filename=root.imagenet.loader.original_labels_fnme,
            count_samples_filename=root.imagenet.loader.count_samples_filename,
            matrixes_filename=root.imagenet.loader.matrixes_filename,
            samples_filename=root.imagenet.loader.samples_filename,
            channels=root.imagenet.loader.channels,
            mirror=root.imagenet.loader.mirror,
        )
        self.loader.link_from(init_unit)


def run(load, main):
    load(ImagenetWorkflow,
         decision_config=root.imagenet.decision,
         snapshotter_config=root.imagenet.snapshotter,
         layers=root.imagenet.layers,
         loss_function=root.imagenet.loss_function)
    main()
