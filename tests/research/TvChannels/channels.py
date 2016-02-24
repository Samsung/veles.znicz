# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 14, 2014

Model created for object recognition (logotypes of TV channels).
Dataset - Channels. Self-constructing Model. It means that Model can change for
any Model (Convolutional, Fully connected, different parameters)
in configuration file.

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
from veles.loader import TEST
from veles.znicz.evaluator import EvaluatorSoftmax

from veles.config import root
from veles.znicz.standard_workflow import StandardWorkflow


class EvaluatorChannels(EvaluatorSoftmax):
    MAPPING = "evaluator_softmax_channels"

    def get_metric_values(self):
        if self.testing:
            output_labels = {}
            for index, labels in enumerate(self.merged_output[:]):
                max_value = 0
                for label_index, value in enumerate(labels):
                    if value >= max_value:
                        max_value = value
                        max_index = label_index
                key = self.class_keys[TEST][index]
                if key not in output_labels:
                    output_labels[key] = [
                        (self.labels_mapping[max_index], max_value)]
                else:
                    output_labels[key].append(
                        (self.labels_mapping[max_index], max_value))
            result = {}
            for key, value in output_labels.items():
                total_max_value = 0
                index_bbox = - 1
                got_smth = False
                for label, max_val in value:
                    index_bbox += 1
                    if total_max_value < max_val and label != "None":
                        total_max_value = max_val
                        total_max_label = label
                        total_max_bbox = index_bbox
                        got_smth = True
                if got_smth:
                    result[key] = (total_max_label, total_max_bbox)
                else:
                    result[key] = ("no channel", None)
            return {"Output": result}
        return {}


class ChannelsWorkflow(StandardWorkflow):
    """
    Model created for object recognition (logotypes of TV channels).
Dataset - Channels. Self-constructing Model. It means that Model can change for
any Model (Convolutional, Fully connected, different parameters)
in configuration file.
    """
    def link_evaluator(self, *parents):
        self.evaluator = EvaluatorChannels(self)
        self.evaluator.link_from(*parents)\
            .link_attrs(self.forwards[-1], "output") \
            .link_attrs(self.loader,
                        ("batch_size", "minibatch_size"),
                        ("labels", "minibatch_labels"),
                        ("max_samples_per_epoch", "total_samples"),
                        "class_lengths", ("offset", "minibatch_offset"))
        if hasattr(self.loader, "reversed_labels_mapping"):
            self.evaluator.link_attrs(
                self.loader, ("labels_mapping", "reversed_labels_mapping"))
        if hasattr(self.loader, "class_keys"):
            self.evaluator.link_attrs(self.loader, "class_keys")
        self.evaluator.link_attrs(self.forwards[-1], "max_idx")
        return self.evaluator

    def create_workflow(self):
        self.link_downloader(self.start_point)
        self.link_repeater(self.downloader)
        self.link_loader(self.repeater)
        self.link_forwards(("input", "minibatch_data"), self.loader)
        self.link_evaluator(self.forwards[-1])
        self.link_decision(self.evaluator)
        end_units = [link(self.decision) for link in (
            self.link_snapshotter, self.link_error_plotter,
            self.link_conf_matrix_plotter)]
        last_gd = self.link_gds(*end_units)
        self.link_loop(last_gd)
        self.link_end_point(last_gd)


def run(load, main):
    load(ChannelsWorkflow,
         decision_config=root.channels.decision,
         snapshotter_config=root.channels.snapshotter,
         image_saver_config=root.channels.image_saver,
         loader_config=root.channels.loader,
         downloader_config=root.channels.downloader,
         layers=root.channels.layers,
         loader_name=root.channels.loader_name,
         evaluator_name=root.channels.evaluator_name,
         decision_name=root.channels.decision_name)
    main()
