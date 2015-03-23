#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.MnistAE.mnist_ae as mnist_ae


class TestMnistAE(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.mnist_ae.update({
            "all2all": {"weights_stddev": 0.05},
            "decision": {"fail_iterations": 20,
                         "max_epochs": 3},
            "snapshotter": {"prefix": "mnist", "time_interval": 0,
                            "interval": 3, "compress": ""},
            "loader": {"minibatch_size": 100, "force_cpu": False,
                       "normalization_type": "linear"},
            "learning_rate": 0.000001,
            "weights_decay": 0.00005,
            "gradient_moment": 0.00001,
            "weights_plotter": {"limit": 16},
            "pooling": {"kx": 3, "ky": 3, "sliding": (2, 2)},
            "include_bias": False,
            "unsafe_padding": True,
            "n_kernels": 5,
            "kx": 5,
            "ky": 5})

    def init_wf(self, workflow, device, snapshot):
        workflow.initialize(device=device, snapshot=snapshot)

    def check_write_error_rate(self, workflow, error):
        err = workflow.decision.epoch_metrics[1][0]
        self.assertLess(err, error)
        self.assertEqual(
            workflow.decision.max_epochs, workflow.loader.epoch_number)

    def init_and_run(self, device, snapshot):
        workflow = mnist_ae.MnistAEWorkflow(
            self.parent,
            layers=root.mnist_ae.layers)
        self.init_wf(workflow, device, snapshot)
        workflow.run()
        return workflow

    mse = {"ocl": (0.96093162, 0.9606072, 0.96072854),
           "cuda": (0.9612299373, 0.9606072, 0.96101219)}

    @timeout(1500)
    @multi_device()
    def test_mnist_ae_gpu(self):
        self.info("Will run workflow on double precision")
        root.common.update({"precision_type": "double"})
        mse = self.mse[self.device.backend_name]

        # Test workflow
        workflow = self.init_and_run(self.device, False)
        self.check_write_error_rate(workflow, mse[0])

        file_name = workflow.snapshotter.file_name

        # Test loading from snapshot
        self.info("Will load workflow from snapshot: %s" % file_name)

        workflow_from_snapshot = Snapshotter.import_(file_name)
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 6
        workflow_from_snapshot.decision.complete <<= False

        self.init_wf(workflow_from_snapshot, self.device, True)
        workflow_from_snapshot.run()
        self.check_write_error_rate(workflow_from_snapshot, mse[1])

        self.info("Will run workflow with float and ocl backend")

        root.common.update({"precision_type": "float"})

        # Test workflow with ocl and float
        workflow = self.init_and_run(self.device, False)
        self.check_write_error_rate(workflow, mse[2])

        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
