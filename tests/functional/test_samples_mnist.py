#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
import veles.znicz.samples.MnistSimple.mnist as mnist
from veles.znicz.tests.functional import StandardTest


class TestSamplesMnist(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.mnist.update({
            "all2all": {"weights_stddev": 0.05},
            "decision": {"fail_iterations": (0),
                         "snapshot_prefix": "samples_mnist_test"},
            "loader": {"minibatch_size": 88, "normalization_type": "linear"},
            "learning_rate": 0.028557478339518444,
            "weights_decay": 0.00012315096341168246,
            "layers": [364, 10],
            "factor_ortho": 0.001})

    @timeout(300)
    @multi_device()
    def test_samples_mnist(self):
        self.info("Will test mnist fully connected workflow from samples")

        workflow = mnist.MnistWorkflow(self.parent, layers=root.mnist.layers)
        workflow.decision.max_epochs = 2
        workflow.snapshotter.time_interval = 0
        workflow.snapshotter.interval = 2
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(
            device=self.device,
            learning_rate=root.mnist.learning_rate,
            weights_decay=root.mnist.weights_decay,
            snapshot=False)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        file_name = workflow.snapshotter.file_name

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 817)
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
            learning_rate=root.mnist.learning_rate,
            weights_decay=root.mnist.weights_decay,
            snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        self.assertEqual(err, 650)
        self.assertEqual(5, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
