# -*- coding: utf-8 -*-
# !/usr/bin/python3 -O
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on April 9, 2015

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


from veles.tests import timeout, multi_device
from veles.znicz.standard_workflow import StandardWorkflow
from veles.znicz.tests.functional import StandardTest


class TestStandardWorkflow(StandardTest):
    def set_parameters_0(self):
        mcdnnic_topology = "12x256x256-32C4-MP2-64C4-MP3-32N-4N"
        mcdnnic_parameters = {"<-": {"learning_rate": 0.03}}
        real_layers = [{'<-': {'learning_rate': 0.03},
                        '->': {'ky': 4, 'kx': 4, 'n_kernels': 32},
                        'type': 'conv'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'ky': 2, 'kx': 2}, 'type': 'max_pooling'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'ky': 4, 'kx': 4, 'n_kernels': 64},
                        'type': 'conv'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'ky': 3, 'kx': 3},
                        'type': 'max_pooling'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'output_sample_shape': 32},
                        'type': 'all2all'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'output_sample_shape': 4}, 'type': 'softmax'}]
        real_loader_params = {'scale': (256, 256), 'minibatch_size': 12}
        kwargs = {}
        return (mcdnnic_topology, mcdnnic_parameters,
                real_layers, real_loader_params, kwargs)

    def set_parameters_1(self):
        mcdnnic_topology = ""
        mcdnnic_parameters = {"<-": {"learning_rate": 0.03}}
        real_layers = [{}]
        real_loader_params = {}
        kwargs = {}
        return (mcdnnic_topology, mcdnnic_parameters,
                real_layers, real_loader_params, kwargs)

    def set_parameters_2(self):
        mcdnnic_topology = "1x56x34-32C4-MP2-32N-600N-11N-4N"
        mcdnnic_parameters = {}
        real_layers = [{'<-': {},
                        '->': {'ky': 4, 'kx': 4, 'n_kernels': 32},
                        'type': 'conv'},
                       {'<-': {}, '->': {'ky': 2, 'kx': 2},
                        'type': 'max_pooling'},
                       {'<-': {}, '->': {'output_sample_shape': 32},
                        'type': 'all2all'},
                       {'<-': {}, '->': {'output_sample_shape': 600},
                        'type': 'all2all'},
                       {'<-': {}, '->': {'output_sample_shape': 11},
                        'type': 'all2all'},
                       {'<-': {}, '->': {'output_sample_shape': 4},
                        'type': 'softmax'}]
        real_loader_params = {'scale': (56, 34), 'minibatch_size': 1,
                              'train_path': '/tmp/file.txt',
                              'normalization_type': 'mean_disp'}
        kwargs = {'minibatch_size': 12, 'train_path': '/tmp/file.txt',
                  'normalization_type': 'mean_disp'}
        return (mcdnnic_topology, mcdnnic_parameters,
                real_layers, real_loader_params, kwargs)

    def set_parameters_3(self):
        mcdnnic_topology = "AB..hj_--cccx7777-6868"
        mcdnnic_parameters = {
            '->': {"weights_filling": "uniform", "weights_stddev": 0.05,
                   "bias_filling": "uniform"},
            "<-": {"learning_rate": 0.03}}
        real_layers = [{}]
        real_loader_params = {}
        kwargs = {}
        return (mcdnnic_topology, mcdnnic_parameters,
                real_layers, real_loader_params, kwargs)

    def set_parameters_4(self):
        mcdnnic_topology = "1x56x34-32C4-MP2-32N-600N-11N-4N"
        mcdnnic_parameters = {
            '->': {"weights_filling": "uniform", "weights_stddev": 0.05,
                   "bias_filling": "uniform"},
            "<-": {"learning_rate": 0.03}}
        real_layers = [{'<-': {'learning_rate': 0.03},
                        '->': {'bias_filling': 'uniform',
                               'weights_stddev': 0.05,
                               'weights_filling': 'uniform', 'ky': 4, 'kx': 4,
                               'n_kernels': 32}, 'type': 'conv'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'weights_stddev': 0.05, 'ky': 2, 'kx': 2,
                               'weights_filling': 'uniform',
                               'bias_filling': 'uniform'},
                        'type': 'max_pooling'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'weights_stddev': 0.05,
                               'bias_filling': 'uniform',
                               'weights_filling': 'uniform',
                               'output_sample_shape': 32},
                        'type': 'all2all'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'weights_stddev': 0.05,
                               'bias_filling': 'uniform',
                               'weights_filling': 'uniform',
                               'output_sample_shape': 600}, 'type': 'all2all'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'weights_stddev': 0.05,
                               'bias_filling': 'uniform',
                               'weights_filling': 'uniform',
                               'output_sample_shape': 11}, 'type': 'all2all'},
                       {'<-': {'learning_rate': 0.03},
                        '->': {'weights_stddev': 0.05,
                               'bias_filling': 'uniform',
                               'weights_filling': 'uniform',
                               'output_sample_shape': 4}, 'type': 'softmax'}]
        real_loader_params = {'scale': (56, 34), 'minibatch_size': 1}
        kwargs = {}
        return (mcdnnic_topology, mcdnnic_parameters,
                real_layers, real_loader_params, kwargs)

    def run_and_assert(self, mcdnnic_topology, mcdnnic_parameters, real_layers,
                       real_loader_params, kwargs):
        workflow = StandardWorkflow(
            self.parent,
            loader_name="full_batch_auto_label_file_image",
            loader_config={},
            loss_function="softmax",
            mcdnnic_topology=mcdnnic_topology,
            mcdnnic_parameters=mcdnnic_parameters)

        generated_layers = workflow._get_layers_from_mcdnnic(
            workflow.mcdnnic_topology)

        generated_params = workflow._update_loader_kwargs_from_mcdnnic(
            kwargs, workflow.mcdnnic_topology)
        self.assertEqual(real_layers, generated_layers)
        self.assertEqual(real_loader_params, generated_params)

    @timeout(50)
    @multi_device()
    def test_mcdnnic_topology_standard_workflow(self):
        for i in range(5):
            self.info("Will test number %s" % i)
            (mcdnnic_topology, mcdnnic_parameters, real_layers,
             real_loader_params, kwargs) = getattr(
                self, "set_parameters_%s" % i)()
            if i == 1 or i == 3:
                with self.assertRaises(ValueError):
                    self.run_and_assert(
                        mcdnnic_topology, mcdnnic_parameters,
                        real_layers, real_loader_params, kwargs)
            else:
                self.run_and_assert(
                    mcdnnic_topology, mcdnnic_parameters,
                    real_layers, real_loader_params, kwargs)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
