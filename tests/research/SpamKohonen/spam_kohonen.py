#!/usr/bin/python3 -O
"""
Created on May 12, 2014

Kohonen Spam detection on Lee Man Ha dataset.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""

import json
import lzma
import os

import numpy
import six
from zope.interface import implementer

from veles.config import root
from veles.external.progressbar import ProgressBar
from veles.interaction import Shell
import veles.units as units
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.kohonen as kohonen
from veles.loader import IFullBatchLoader
from veles import plotting_units, loader


spam_dir = os.path.dirname(__file__)

root.spam_kohonen.update({
    "forward": {"shape": (8, 8),
                "weights_stddev": 0.05,
                "weights_filling": "uniform"},
    "decision": {"snapshot_prefix": "spam_kohonen",
                 "epochs": 5},
    "loader": {"minibatch_size": 60, "force_cpu": True,
               "file": os.path.join(spam_dir, "data.txt.xz")},
    "train": {"gradient_decay": lambda t: 0.001 / (1.0 + t * 0.00001),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.00001)},
    "exporter": {"file": "weights.txt"}})
root.spam_kohonen.loader.validation_ratio = 0.0


@implementer(IFullBatchLoader)
class SpamKohonenLoader(loader.FullBatchLoader):
    def __init__(self, workflow, **kwargs):
        kwargs["normalization_type"] = "pointwise"
        super(SpamKohonenLoader, self).__init__(workflow, **kwargs)
        self.has_ids = kwargs.get("ids", False)
        self.has_classes = kwargs.get("classes", True)
        self.lemmas_map = 0
        self.labels_mapping = []
        self.samples_by_label = []
        self.ids = []

    def load_data(self):
        """Here we will load spam data.
        """
        file_name = root.spam_kohonen.loader.file
        if os.path.splitext(file_name)[1] == '.xz':
            self.info("Unpacking %s...", root.spam_kohonen.loader.file)
            if six.PY3:
                with lzma.open(root.spam_kohonen.loader.file, "r") as fin:
                    lines = fin.readlines()
            else:
                fin = lzma.LZMAFile(root.spam_kohonen.loader.file, "r")
                lines = fin.readlines()
                fin.close()
        else:
            self.info("Reading %s...", root.spam_kohonen.loader.file)
            with open(root.spam_kohonen.loader.file, "rb") as fin:
                lines = fin.readlines()

        self.info("Parsing the data...")
        lemmas = set()
        data = []
        del self.ids[:]
        labels = []
        avglength = 0
        for line in ProgressBar(term_width=17)(lines):
            fields = line.split(b' ')
            offset = 0
            if self.has_ids:
                self.ids.append(fields[offset].decode('charmap'))
                offset += 1
            if self.has_classes:
                label = int(fields[offset])
                labels.append(label)
                offset += 1
            data.append([])
            for field in fields[offset:-1]:
                lemma, weight = field.split(b':')
                lemma = int(lemma)
                weight = float(weight)
                lemmas.add(lemma)
                data[-1].append((lemma, weight))
            avglength += len(data[-1])

        self.info("Initializing...")
        avglength //= len(lines)
        if self.has_classes:
            distinct_labels = set(labels)
        else:
            distinct_labels = {0}
        del self.labels_mapping[:]
        self.labels_mapping.extend(sorted(distinct_labels))
        reverse_label_mapping = {l: i for i, l
                                 in enumerate(self.labels_mapping)}
        self.lemmas_map = sorted(lemmas)
        lemma_indices = {v: i for i, v in enumerate(self.lemmas_map)}
        self.original_labels.mem = numpy.zeros([len(lines)], dtype=numpy.int32)
        del self.samples_by_label[:]
        self.samples_by_label.extend([set() for _ in distinct_labels])
        self.original_data.mem = numpy.zeros([len(lines), len(lemmas)],
                                             dtype=numpy.float32)
        for index, sample in enumerate(ProgressBar(term_width=17)(data)):
            if self.has_classes:
                label = reverse_label_mapping[labels[index]]
            else:
                label = 0
            self.original_labels.mem[index] = label
            self.samples_by_label[label].add(index)
            for lemma in sample:
                self.original_data.mem[index,
                                       lemma_indices[lemma[0]]] = lemma[1]

        self.validation_ratio = root.spam_kohonen.loader.validation_ratio
        self.class_lengths[loader.TEST] = 0
        self.class_lengths[loader.VALID] = int(self.validation_ratio *
                                               len(lines))
        self.class_lengths[loader.TRAIN] = len(lines) - self.class_lengths[1]
        if self.class_lengths[loader.VALID] > 0:
            self.resize_validation()
        self.info("Samples: %d, labels: %d, lemmas: %d, "
                  "average feature vector length: %d", len(lines),
                  len(distinct_labels), len(lemmas), avglength)

    def initialize(self, device, **kwargs):
        super(SpamKohonenLoader, self).initialize(device=device, **kwargs)
        if self.class_lengths[loader.VALID] > 0:
            v = self.original_data.mem[:self.class_lengths[loader.VALID]]
            self.info("Range after normalization: validation: [%.6f, %.6f]",
                      v.min(), v.max())
        v = self.original_data.mem[self.class_lengths[loader.VALID]:]
        self.info("Range after normalization: train: [%.6f, %.6f]",
                  v.min(), v.max())


@implementer(units.IUnit)
class ResultsExporter(units.Unit):
    """Exports Kohonen network weights to a text file.
    """
    def __init__(self, workflow, file_name, **kwargs):
        super(ResultsExporter, self).__init__(workflow, **kwargs)
        self.file_name = file_name
        self.demand("shuffled_indices", "total", "ids")

    def initialize(self, **kwargs):
        pass

    def run(self):
        self.total.map_read()

        self.info("Working...")
        if len(self.ids) == 0:
            classified = numpy.zeros(len(self.shuffled_indices),
                                     dtype=numpy.int32)
            for i in range(self.total.mem.size):
                classified[self.shuffled_indices[i]] = self.total[i]
            numpy.savetxt(self.file_name, classified, fmt='%d')
        else:
            classified = {}
            for i in range(self.total.mem.size):
                classified[self.ids[self.shuffled_indices[i]]] = \
                    int(self.total[i])
            with open(self.file_name, "w") as fout:
                json.dump(classified, fout, indent=4)
        self.info("Exported the classified data to %s", self.file_name)


class SpamKohonenWorkflow(nn_units.NNWorkflow):
    """Workflow for Kohonen Spam Detection.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "Kohonen Spam")
        super(SpamKohonenWorkflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = SpamKohonenLoader(
            self, name="Kohonen Spam fullbatch loader",
            minibatch_size=root.spam_kohonen.loader.minibatch_size,
            force_cpu=root.spam_kohonen.loader.force_cpu,
            ids=root.spam_kohonen.loader.ids,
            classes=root.spam_kohonen.loader.classes)
        self.loader.link_from(self.repeater)

        # Kohonen training layer
        self.trainer = kohonen.KohonenTrainer(
            self,
            shape=root.spam_kohonen.forward.shape,
            weights_filling=root.spam_kohonen.forward.weights_filling,
            weights_stddev=root.spam_kohonen.forward.weights_stddev,
            gradient_decay=root.spam_kohonen.train.gradient_decay,
            radius_decay=root.spam_kohonen.train.radius_decay)
        self.trainer.link_from(self.loader)
        self.trainer.link_attrs(self.loader, ("input", "minibatch_data"))

        self.forward = kohonen.KohonenForward(self, total=True)
        self.forward.link_from(self.trainer)
        self.forward.link_attrs(self.loader, ("input", "minibatch_data"),
                                "minibatch_offset", "minibatch_size",
                                ("batch_size", "total_samples"))
        self.forward.link_attrs(self.trainer, "weights", "argmins")

        if root.spam_kohonen.loader.classes:
            self.validator = kohonen.KohonenValidator(self)
            self.validator.link_attrs(self.trainer, "shape")
            self.validator.link_attrs(self.forward, ("input", "output"))
            self.validator.link_attrs(self.loader, "minibatch_indices",
                                                   "minibatch_size",
                                                   "samples_by_label")
            self.validator.link_from(self.forward)

        # Loop decision
        self.decision = kohonen.KohonenDecision(
            self, max_epochs=root.spam_kohonen.decision.epochs)
        self.decision.link_from(
            self.validator if root.spam_kohonen.loader.classes
            else self.forward)

        self.decision.link_attrs(self.loader, "minibatch_class",
                                              "last_minibatch",
                                              "class_lengths",
                                              "epoch_ended",
                                              "epoch_number")
        self.decision.link_attrs(self.trainer, "weights", "winners")

        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.loader.epoch_ended

        self.repeater.link_from(self.ipython)

        self.exporter = ResultsExporter(self, root.spam_kohonen.exporter.file)
        self.exporter.link_from(self.decision)
        self.exporter.total = self.forward.total
        self.exporter.link_attrs(self.loader, "shuffled_indices", "ids")
        self.exporter.gate_block = ~self.decision.complete

        self.end_point.link_from(self.exporter)

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plotters = [nn_plotting_units.KohonenHits(self),
                         nn_plotting_units.KohonenNeighborMap(self),
                         plotting_units.AccumulatingPlotter(
                             self, name="Weights epoch difference",
                             plot_style="g-", redraw_plot=True,
                             clear_plot=True)]
        if root.spam_kohonen.loader.classes:
            self.plotters.append(
                nn_plotting_units.KohonenValidationResults(self))
        self.plotters[0].link_attrs(self.trainer, "shape")
        self.plotters[0].link_attrs(self.decision, ("input", "winners_mem"))
        self.plotters[0].link_from(self.decision)
        self.plotters[0].gate_block = ~self.loader.epoch_ended
        self.plotters[1].link_attrs(self.trainer, "shape")
        self.plotters[1].link_attrs(self.decision, ("input", "weights_mem"))
        self.plotters[1].link_from(self.decision)
        self.plotters[1].gate_block = ~self.loader.epoch_ended
        self.plotters[2].link_attrs(self.trainer, "shape")
        self.plotters[2].link_attrs(self.decision, ("input", "weights_diff"))
        self.plotters[2].link_from(self.decision)
        self.plotters[2].gate_block = ~self.loader.epoch_ended
        if root.spam_kohonen.loader.classes:
            self.plotters[3].link_attrs(self.trainer, "shape")
            self.plotters[3].link_attrs(self.validator, "result", "fitness",
                                        "fitness_by_label",
                                        "fitness_by_neuron")
            self.plotters[3].link_attrs(self.decision,
                                        ("input", "winners_mem"))
            self.plotters[3].link_from(self.decision)
            self.plotters[3].gate_block = ~self.loader.epoch_ended

    def initialize(self, device, **kwargs):
        return super(SpamKohonenWorkflow, self).initialize(device=device)


def run(load, main):
    load(SpamKohonenWorkflow)
    main()
