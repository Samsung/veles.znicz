#!/usr/bin/python3.3 -O
"""
Created on May 12, 2014

Kohonen Spam detection on Lee Man Ha dataset.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import lzma
import numpy
import six
import os
from zope.interface import implementer

from veles.config import root
from veles.external.progressbar import ProgressBar
from veles.interaction import Shell
import veles.formats as formats
import veles.units as units
import veles.znicz.loader as loader
import veles.znicz.nn_plotting_units as nn_plotting_units
import veles.znicz.nn_units as nn_units
import veles.znicz.kohonen as kohonen
from veles.znicz.loader import IFullBatchLoader


spam_dir = os.path.join(os.path.dirname(__file__), "spam")

root.defaults = {
    "forward": {"shape": (8, 8),
                "weights_stddev": 0.05,
                "weights_filling": "uniform"},
    "decision": {"snapshot_prefix": "spam_kohonen",
                 "epochs": 5},
    "loader": {"minibatch_size": 60,
               "file": os.path.join(spam_dir, "data.txt.xz")},
    "train": {"gradient_decay": lambda t: 0.001 / (1.0 + t * 0.00001),
              "radius_decay": lambda t: 1.0 / (1.0 + t * 0.00001)},
    "exporter": {"file": "weights.txt"}}
root.loader.validation_ratio = 0


@implementer(IFullBatchLoader)
class Loader(loader.FullBatchLoader):
    def __init__(self, workflow, **kwargs):
        super(Loader, self).__init__(workflow, **kwargs)
        self.lemmas_map = 0

    def load_data(self):
        """Here we will load spam data.
        """
        file_name = root.loader.file
        if os.path.splitext(file_name)[1] == '.xz':
            self.info("Unpacking %s...", root.loader.file)
            if six.PY3:
                with lzma.open(root.loader.file, "r") as fin:
                    lines = fin.readlines()
            else:
                fin = lzma.LZMAFile(root.loader.file, "r")
                lines = fin.readlines()
                fin.close()
        else:
            self.info("Reading %s...", root.loader.file)
            with open(root.loader.file, "rb") as fin:
                lines = fin.readlines()

        self.info("Parsing the data...")
        progress = ProgressBar(maxval=len(lines), term_width=17)
        progress.start()
        lemmas = set()
        data = []
        spam_count = 0
        avglength = 0
        for line in lines:
            fields = line.split(b' ')
            data.append((int(fields[0]), []))
            if data[-1][0] == 1:
                spam_count += 1
            for field in fields[1:-1]:
                lemma, weight = field.split(b':')
                lemma = int(lemma)
                weight = float(weight)
                lemmas.add(lemma)
                data[-1][1].append((lemma, weight))
            avglength += len(data[-1][1])
            progress.inc()
        progress.finish()
        avglength //= len(lines)

        self.info("Initializing...")
        progress = ProgressBar(maxval=len(data), term_width=17)
        progress.start()
        self.lemmas_map = sorted(lemmas)
        lemma_indices = {v: i for i, v in enumerate(self.lemmas_map)}
        self.original_labels = numpy.zeros([len(lines)], dtype=numpy.int32)
        self.original_data = numpy.zeros([len(lines), len(lemmas)],
                                         dtype=numpy.float32)
        for index, sample in enumerate(data):
            self.original_labels[index] = sample[0]
            for lemma in sample[1]:
                self.original_data[index, lemma_indices[lemma[0]]] = lemma[1]
            progress.inc()
        progress.finish()

        self.validation_ratio = root.loader.validation_ratio
        self.class_lengths[loader.TEST] = 0
        self.class_lengths[loader.VALID] = self.validation_ratio * len(lines)
        self.class_lengths[loader.TRAIN] = len(lines) - self.class_lengths[1]
        if self.class_lengths[loader.VALID] > 0:
            self.extract_validation_from_train()
        self.info("Samples: %d (spam: %d), lemmas: %d, "
                  "average feature vector length: %d", len(lines), spam_count,
                  len(lemmas), avglength)
        self.info("Normalizing...")
        self.IMul, self.IAdd = formats.normalize_pointwise(
            self.original_data[self.class_lengths[loader.VALID]:])
        self.original_data *= self.IMul
        self.original_data += self.IAdd
        if self.class_lengths[loader.VALID] > 0:
            v = self.original_data[:self.class_lengths[loader.VALID]]
            self.info("Range after normalization: validation: [%.6f, %.6f]",
                      v.min(), v.max())
        v = self.original_data[self.class_lengths[loader.VALID]:]
        self.info("Range after normalization: train: [%.6f, %.6f]",
                  v.min(), v.max())


@implementer(units.IUnit)
class ResultsExporter(units.Unit):
    """Exports Kohonen network weights to a text file.
    """
    def __init__(self, workflow, file_name, **kwargs):
        super(ResultsExporter, self).__init__(workflow, **kwargs)
        self.total = None
        self.shuffled_indices = None
        self.file_name = file_name

    def initialize(self, **kwargs):
        pass

    def run(self):
        self.total.map_read()

        classified = numpy.zeros(len(self.shuffled_indices), dtype=numpy.int32)
        for i in range(self.total.mem.size):
            classified[self.shuffled_indices[i]] = self.total[i]
        numpy.savetxt(self.file_name, classified, fmt='%d')
        self.info("Exported the classified data to %s", self.file_name)


class Workflow(nn_units.NNWorkflow):
    """Workflow for Kohonen Spam Detection.
    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "Kohonen Spam")
        super(Workflow, self).__init__(workflow, **kwargs)

        self.repeater.link_from(self.start_point)

        self.loader = Loader(self, name="Kohonen Spam fullbatch loader",
                             minibatch_size=root.loader.minibatch_size)
        self.loader.link_from(self.repeater)

        # Kohonen training layer
        self.trainer = kohonen.KohonenTrainer(
            self,
            shape=root.forward.shape,
            weights_filling=root.forward.weights_filling,
            weights_stddev=root.forward.weights_stddev,
            gradient_decay=root.train.gradient_decay,
            radius_decay=root.train.radius_decay)
        self.trainer.link_from(self.loader)
        self.trainer.link_attrs(self.loader, ("input", "minibatch_data"))

        self.forward = kohonen.KohonenForward(self, total=True)
        self.forward.link_from(self.trainer)
        self.forward.link_attrs(self.loader, ("input", "minibatch_data"),
                                "minibatch_offset", "minibatch_size",
                                ("batch_size", "total_samples"))
        self.forward.link_attrs(self.trainer, "weights", "argmins")

        # Loop decision
        self.decision = kohonen.KohonenDecision(
            self, max_epochs=root.decision.epochs)
        self.decision.link_from(self.forward)
        self.decision.link_attrs(self.loader, "minibatch_class",
                                              "no_more_minibatches_left",
                                              "class_lengths")
        self.decision.link_attrs(self.trainer, "weights", "winners")
        self.trainer.epoch_ended = self.decision.epoch_ended

        self.ipython = Shell(self)
        self.ipython.link_from(self.decision)
        self.ipython.gate_skip = ~self.decision.epoch_ended

        self.repeater.link_from(self.ipython)

        self.exporter = ResultsExporter(self, root.exporter.file)
        self.exporter.link_from(self.decision)
        self.exporter.total = self.forward.total
        self.exporter.link_attrs(self.loader, "shuffled_indices")
        self.exporter.gate_block = ~self.decision.complete

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plotters = [nn_plotting_units.KohonenHits(self),
                         nn_plotting_units.KohonenNeighborMap(self)]
        self.plotters[0].link_attrs(self.trainer, "shape")
        self.plotters[0].input = self.decision.winners_copy
        self.plotters[0].link_from(self.decision)
        self.plotters[0].gate_block = ~self.decision.epoch_ended
        self.plotters[1].link_attrs(self.trainer, "shape")
        self.plotters[1].input = self.decision.weights_copy
        self.plotters[1].link_from(self.decision)
        self.plotters[1].gate_block = ~self.decision.epoch_ended

    def initialize(self, device):
        return super(Workflow, self).initialize(device=device)


def run(load, main):
    load(Workflow)
    main()
