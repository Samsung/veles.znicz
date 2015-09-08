#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on August 12, 2015

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


from codecs import getreader
import json
import os
import tarfile
from tempfile import NamedTemporaryFile
import zipfile

from veles.backends import NumpyDevice
from veles.config import root
from veles.genetics import Range, fix_config
from veles.znicz.all2all import All2AllTanh, All2AllSoftmax
import veles.znicz.samples.MNIST.mnist as mnist_all2all
from veles.znicz.tests.functional import StandardTest


class TestPackageExport(StandardTest):
    DEVICE = NumpyDevice

    @classmethod
    def setUpClass(cls):
        root.mnistr.update({
            "loss_function": "softmax",
            "loader_name": "mnist_loader",
            "lr_adjuster": {"do": False},
            "decision": {"max_epochs": 1},
            "weights_plotter": {"limit": 0},
            "snapshotter": {"interval": 100},
            "loader": {"minibatch_size": Range(60, 1, 1000),
                       "force_numpy": False,
                       "normalization_type": "linear",
                       "data_path":
                       os.path.join(root.common.dirs.datasets, "MNIST")},
            "layers": [{"type": "all2all_tanh",
                        "->": {"output_sample_shape": Range(100, 10, 500),
                               "weights_filling": "uniform",
                               "weights_stddev": Range(0.05, 0.0001, 0.1),
                               "bias_filling": "uniform",
                               "bias_stddev": Range(0.05, 0.0001, 0.1)},
                        "<-": {"learning_rate": Range(0.03, 0.0001, 0.9),
                               "weights_decay": Range(0.0, 0.0, 0.9),
                               "learning_rate_bias": Range(0.03, 0.0001, 0.9),
                               "weights_decay_bias": Range(0.0, 0.0, 0.9),
                               "gradient_moment": Range(0.0, 0.0, 0.95),
                               "gradient_moment_bias": Range(0.0, 0.0, 0.95),
                               "factor_ortho": Range(0.001, 0.0, 0.1)}},
                       {"type": "softmax",
                        "->": {"output_sample_shape": 10,
                               "weights_filling": "uniform",
                               "weights_stddev": Range(0.05, 0.0001, 0.1),
                               "bias_filling": "uniform",
                               "bias_stddev": Range(0.05, 0.0001, 0.1)},
                        "<-": {"learning_rate": Range(0.03, 0.0001, 0.9),
                               "learning_rate_bias": Range(0.03, 0.0001, 0.9),
                               "weights_decay": Range(0.0, 0.0, 0.95),
                               "weights_decay_bias": Range(0.0, 0.0, 0.95),
                               "gradient_moment": Range(0.0, 0.0, 0.95),
                               "gradient_moment_bias": Range(0.0, .0, 0.95)}}]}
        )
        fix_config(root)
        root.common.disable.snapshotting = True

    def test_package_export_mnist_all2all(self):
        workflow = mnist_all2all.MnistWorkflow(
            self.parent,
            decision_config=root.mnistr.decision,
            snapshotter_config=root.mnistr.snapshotter,
            loader_name=root.mnistr.loader_name,
            loader_config=root.mnistr.loader,
            layers=root.mnistr.layers,
            loss_function=root.mnistr.loss_function)
        workflow.initialize(device=self.device)
        workflow.run()

        def validate(contents, files):
            unit0 = contents["units"][0]
            self.assertEqual(unit0["class"]["uuid"], All2AllTanh.__id__)
            self.assertEqual(contents["units"][1]["class"]["uuid"],
                             All2AllSoftmax.__id__)
            for unit in contents["units"]:
                for attr in ("bias", "weights"):
                    self.assertTrue("%s.npy" % unit["data"][attr][1:] in files)
            self.assertTrue(1 in unit0["links"])

        with NamedTemporaryFile(suffix="-veles-test-package.tar.gz") as fpkg:
            workflow.package_export(fpkg.name, archive_format="tgz")
            fpkg.seek(0)
            with tarfile.open(fileobj=fpkg, mode="r:gz") as tar:
                contents_ = json.load(getreader("utf-8")(
                    tar.extractfile("contents.json")))
                files_ = tar.getnames()
            validate(contents_, files_)
        with NamedTemporaryFile(suffix="-veles-test-package.zip") as fpkg:
            workflow.package_export(fpkg.name, archive_format="zip",
                                    precision=16)
            fpkg.seek(0)
            with zipfile.ZipFile(fpkg) as azip:
                contents_ = json.loads(azip.read(
                    "contents.json").decode("utf-8"))
                files_ = azip.namelist()
            validate(contents_, files_)


if __name__ == "__main__":
    StandardTest.main()
