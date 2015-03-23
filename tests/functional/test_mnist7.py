#!/usr/bin/python3 -O
"""
Created on October 13, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.Mnist7.mnist7 as mnist7


class TestMnist7(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.mnist7.update({
            "decision": {"fail_iterations": 25, "max_epochs": 2},
            "snapshotter": {"prefix": "mnist7_test", "interval": 3,
                            "time_interval": 0},
            "loader": {"minibatch_size": 60, "force_cpu": False,
                       "normalization_type": "linear"},
            "learning_rate": 0.0001,
            "weights_decay": 0.00005,
            "layers": [100, 100, 7]})

    @timeout(300)
    def test_mnist7(self):
        self.info("Will test mnist7 workflow")

        workflow = mnist7.Mnist7Workflow(
            self.parent, layers=root.mnist7.layers)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(
            device=self.device,
            learning_rate=root.mnist7.learning_rate,
            weights_decay=root.mnist7.weights_decay,
            snapshot=False)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        file_name = workflow.snapshotter.file_name

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 8990)
        avg_mse = workflow.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.821236, places=5)
        self.assertEqual(2, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = Snapshotter.import_(file_name)
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 5
        workflow_from_snapshot.decision.complete <<= False
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.initialize(
            device=self.device,
            learning_rate=root.mnist7.learning_rate,
            weights_decay=root.mnist7.weights_decay,
            snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        self.assertEqual(err, 8804)
        avg_mse = workflow_from_snapshot.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.759115, places=5)
        self.assertEqual(5, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")


if __name__ == "__main__":
    StandardTest.main()
