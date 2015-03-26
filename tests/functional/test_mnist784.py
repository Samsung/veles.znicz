#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.Mnist784.mnist784 as mnist784


class TestMnist784(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.mnist784.update({
            "decision": {"fail_iterations": 100, "max_epochs": 3},
            "snapshotter": {"prefix": "mnist_784_test", "time_interval": 0,
                            "interval": 4},
            "loader": {"minibatch_size": 100, "normalization_type": "linear",
                       "target_normalization_type": "linear"},
            "weights_plotter": {"limit": 16},
            "learning_rate": 0.00001,
            "weights_decay": 0.00005,
            "layers": [784, 784]})

    def init_wf(self, workflow, device, snapshot):
        workflow.initialize(device=device,
                            learning_rate=root.mnist784.learning_rate,
                            weights_decay=root.mnist784.weights_decay,
                            snapshot=snapshot)

    def check_write_error_rate(self, workflow, mse, error):
        avg_mse = workflow.decision.epoch_metrics[1][0]
        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, error)
        self.assertAlmostEqual(avg_mse, mse, places=6)
        self.assertEqual(
            workflow.decision.max_epochs, workflow.loader.epoch_number)

    def init_and_run(self, device, snapshot):
        workflow = mnist784.Mnist784Workflow(
            self.parent, layers=root.mnist784.layers)
        self.init_wf(workflow, device, snapshot)
        workflow.run()
        return workflow

    mse = ((0.403674, 7978), (0.391974, 7581))

    @timeout(1000)
    @multi_device()
    def test_mnist784_gpu(self):
        self.info("Will run workflow on double precision")
        # Test workflow
        workflow = self.init_and_run(self.device, False)
        self.check_write_error_rate(workflow, *self.mse[0])

        file_name = workflow.snapshotter.file_name

        # Test loading from snapshot
        self.info("Will load workflow from snapshot: %s" % file_name)

        workflow_from_snapshot = Snapshotter.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 5
        workflow_from_snapshot.decision.complete <<= False

        self.init_wf(workflow_from_snapshot, self.device, True)
        workflow_from_snapshot.run()
        self.check_write_error_rate(workflow_from_snapshot, *self.mse[1])

    def test_mnist784_cpu(self):
        self.info("Will run workflow on numpy")
        workflow = self.init_and_run(None, False)
        self.check_write_error_rate(
            workflow, 0.40712743094, 8166)


if __name__ == "__main__":
    StandardTest.main()
