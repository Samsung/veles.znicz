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
import veles.dummy as dummy_workflow


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
        self.w = mnist_ae.MnistAEWorkflow(dummy_workflow.DummyLauncher(),
                                          layers=root.mnist_ae.layers)
        self.init_wf(self.w, device, snapshot)
        self.w.run()

    mse = {"ocl": (0.96093162, 0.9606072, 0.96072854),
           "cuda": (0.9612299373, 0.9606072, 0.96101219)}

    @timeout(1500)
    @multi_device
    def test_mnist_ae_gpu(self):
        self.info("Will run workflow on double precision")
        root.common.update({"precision_type": "double"})
        mse = self.mse[self.device.backend_name]

        # Test workflow
        self.init_and_run(self.device, False)
        self.check_write_error_rate(self.w, mse[0])

        file_name = self.w.snapshotter.file_name

        # Test loading from snapshot
        self.info("Will load workflow from snapshot: %s" % file_name)

        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 6
        self.wf.decision.complete <<= False

        self.init_wf(self.wf, self.device, True)
        self.wf.run()
        self.check_write_error_rate(self.wf, mse[1])

        self.info("Will run workflow with float and ocl backend")

        root.common.update({"precision_type": "float"})

        # Test workflow with ocl and float
        self.init_and_run(self.device, False)
        self.check_write_error_rate(self.w, mse[2])

        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
