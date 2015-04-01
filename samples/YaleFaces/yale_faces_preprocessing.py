#!/usr/bin/python3 -O
'''
Created on Nov 13, 2014

Model was created for face recognition. Database - Yale Faces.
Model - fully-connected Neural Network with SoftMax loss function.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
'''


from veles.config import root
from veles.znicz.standard_workflow import StandardWorkflow


class YaleFacesWorkflow(StandardWorkflow):
    """
    Model was created for face recognition. Database - Yale Faces.
    Model - fully-connected Neural Network with SoftMax loss function.
    """

    def link_end_point(self, init_unit):
        self.end_point.link_from(init_unit)
        self.end_point.gate_block = ~self.loader.train_ended
        self.loader.gate_block = self.loader.train_ended

    def create_workflow(self):
        self.link_repeater(self.start_point)
        self.link_loader(self.repeater)
        self.link_data_saver(self.loader)
        self.link_repeater(self.data_saver)
        self.link_end_point(self.repeater)


def run(load, main):
    load(YaleFacesWorkflow,
         loader_name=root.yalefaces.loader_name,
         data_saver_config=root.yalefaces.datasaver,
         preprocessing=root.yalefaces.preprocessing,
         loader_config=root.yalefaces.loader)
    main()
