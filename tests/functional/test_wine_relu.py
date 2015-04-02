#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.WineRelu.wine_relu as wine_relu


class TestWineRelu(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.wine_relu.update({
            "decision": {"fail_iterations": 250},
            "snapshotter": {"prefix": "wine_relu"},
            "loader": {"minibatch_size": 10},
            "learning_rate": 0.03,
            "weights_decay": 0.0,
            "layers": [10, 3]})

    @timeout(500)
    @multi_device()
    def test_wine_relu(self):
        self.info("Will test wine relu workflow")

        workflow = wine_relu.WineReluWorkflow(
            self.parent,
            layers=root.wine_relu.layers)

        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(
            learning_rate=root.wine_relu.learning_rate,
            weights_decay=root.wine_relu.weights_decay,
            device=self.device, snapshot=False)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)

        epoch = workflow.decision.epoch_number
        self.info("Converged in %d epochs", epoch)
        self.assertEqual(epoch, 161)
        self.info("All Ok")


if __name__ == "__main__":
    StandardTest.main()
