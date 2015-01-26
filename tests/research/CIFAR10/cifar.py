#!/usr/bin/python3 -O
"""
Created on Jul 3, 2013

Model created for object recognition. Dataset - CIFAR10. Self-constructing
Model. It means that Model can change for any Model (Convolutional, Fully
connected, different parameters) in configuration file.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os
import pickle
import six
from zope.interface import implementer

from veles.config import root
import veles.memory as formats
from veles.normalization import normalize_linear
from veles.znicz import loader
from veles.znicz.standard_workflow import StandardWorkflow


@implementer(loader.IFullBatchLoader)
class CifarLoader(loader.FullBatchLoader):
    """Loads Cifar dataset.
    """
    def __init__(self, workflow, **kwargs):
        super(CifarLoader, self).__init__(workflow, **kwargs)
        self.shuffle_limit = kwargs.get("shuffle_limit", 2000000000)

    def shuffle(self):
        if self.shuffle_limit <= 0:
            return
        self.shuffle_limit -= 1
        self.info("Shuffling, remaining limit is %d", self.shuffle_limit)
        super(CifarLoader, self).shuffle()

    def _add_sobel_chan(self):
        """
        Adds 4th channel (Sobel filtered image) to `self.original_data`
        """
        import cv2

        sobel_data = numpy.zeros(shape=self.original_data.shape[:-1],
                                 dtype=numpy.float32)

        for i in range(self.original_data.shape[0]):
            pic = self.original_data[i, :, :, 0:3]
            sobel_x = cv2.Sobel(pic, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(pic, cv2.CV_32F, 0, 1, ksize=3)
            sobel_total = numpy.sqrt(numpy.square(sobel_x) +
                                     numpy.square(sobel_y))
            sobel_gray = cv2.cvtColor(sobel_total, cv2.COLOR_BGR2GRAY)
            normalize_linear(sobel_gray)

            if root.cifar.loader.normalization_type == "mean":
                sobel_data[i, :, :] = (sobel_gray + 1) / 2 * 255
            elif root.cifar.loader.normalization_type == "-128, 128":
                sobel_data[i, :, :] = sobel_gray * 128
            elif root.cifar.loader.normalization_type == "-1, 1":
                sobel_data[i, :, :] = sobel_gray

        sobel_data = sobel_data.reshape(self.original_data.shape[:-1] + (1,))
        numpy.append(self.original_data, sobel_data, axis=3)

    def load_data(self):
        """Here we will load data.
        """
        self.original_data.mem = numpy.zeros([60000, 32, 32, 3],
                                             dtype=numpy.float32)
        self.original_labels.mem = numpy.zeros(60000, dtype=numpy.int32)

        # Load Validation
        with open(root.cifar.data_paths.validation, "rb") as fin:
            if six.PY3:
                vle = pickle.load(fin, encoding='latin1')
            else:
                vle = pickle.load(fin)
        fin.close()
        self.original_data.mem[:10000] = formats.interleave(
            vle["data"].reshape(10000, 3, 32, 32))[:]
        self.original_labels.mem[:10000] = vle["labels"][:]

        # Load Train
        for i in range(1, 6):
            with open(os.path.join(root.cifar.data_paths.train,
                                   ("data_batch_%d" % i)), "rb") as fin:
                if six.PY3:
                    vle = pickle.load(fin, encoding='latin1')
                else:
                    vle = pickle.load(fin)
            self.original_data.mem[i * 10000: (i + 1) * 10000] = (
                formats.interleave(vle["data"].reshape(10000, 3, 32, 32))[:])
            self.original_labels.mem[i * 10000: (i + 1) * 10000] = (
                vle["labels"][:])

        self.class_lengths[0] = 0
        self.class_offsets[0] = 0
        self.class_lengths[1] = 10000
        self.class_offsets[1] = 10000
        self.class_lengths[2] = 50000
        self.class_offsets[2] = 60000

        self.total_samples = self.original_data.shape[0]

        use_sobel = root.cifar.loader.sobel
        if use_sobel:
            self._add_sobel_chan()

        if root.cifar.loader.normalization_type == "mean":
            mean = numpy.mean(self.original_data[10000:], axis=0)
            self.original_data.mem -= mean
            self.info("Validation range: %.6f %.6f %.6f",
                      self.original_data.mem[:10000].min(),
                      self.original_data.mem[:10000].max(),
                      numpy.average(self.original_data.mem[:10000]))
            self.info("Train range: %.6f %.6f %.6f",
                      self.original_data.mem[10000:].min(),
                      self.original_data.mem[10000:].max(),
                      numpy.average(self.original_data.mem[10000:]))
        elif root.cifar.loader.normalization_type == "-1, 1":
            for sample in self.original_data.mem:
                normalize_linear(sample)
        elif root.cifar.loader.normalization_type == "-128, 128":
            for sample in self.original_data.mem:
                normalize_linear(sample)
                sample *= 128
        else:
            raise ValueError("Unsupported normalization type "
                             + str(root.cifar.loader.norm))


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
