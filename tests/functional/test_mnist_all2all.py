#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root
from veles.genetics import Tune, fix_config
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.MNIST.mnist as mnist_all2all


class TestMnistAll2All(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.mnistr.update({
            "loss_function": "softmax",
            "loader_name": "mnist_loader",
            "learning_rate_adjust": {"do": False},
            "decision": {"fail_iterations": 100,
                         "max_epochs": 3},
            "snapshotter": {"prefix": "mnist_all2all_test", "interval": 3},
            "loader": {"minibatch_size": Tune(60, 1, 1000),
                       "force_cpu": False,
                       "normalization_type": "linear"},
            "layers": [{"type": "all2all_tanh",
                        "->": {"output_sample_shape": Tune(100, 10, 500),
                               "weights_filling": "uniform",
                               "weights_stddev": Tune(0.05, 0.0001, 0.1),
                               "bias_filling": "uniform",
                               "bias_stddev": Tune(0.05, 0.0001, 0.1)},
                        "<-": {"learning_rate": Tune(0.03, 0.0001, 0.9),
                               "weights_decay": Tune(0.0, 0.0, 0.9),
                               "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                               "weights_decay_bias": Tune(0.0, 0.0, 0.9),
                               "gradient_moment": Tune(0.0, 0.0, 0.95),
                               "gradient_moment_bias": Tune(0.0, 0.0, 0.95),
                               "factor_ortho": Tune(0.001, 0.0, 0.1)}},
                       {"type": "softmax",
                        "->": {"output_sample_shape": 10,
                               "weights_filling": "uniform",
                               "weights_stddev": Tune(0.05, 0.0001, 0.1),
                               "bias_filling": "uniform",
                               "bias_stddev": Tune(0.05, 0.0001, 0.1)},
                        "<-": {"learning_rate": Tune(0.03, 0.0001, 0.9),
                               "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                               "weights_decay": Tune(0.0, 0.0, 0.95),
                               "weights_decay_bias": Tune(0.0, 0.0, 0.95),
                               "gradient_moment": Tune(0.0, 0.0, 0.95),
                               "gradient_moment_bias": Tune(0.0, 0.0, 0.95)}}]}
        )

    @timeout(300)
    @multi_device()
    def test_mnist_all2all(self):
        self.info("Will test fully connectected mnist workflow")

        fix_config(root)
        workflow = mnist_all2all.MnistWorkflow(
            self.parent,
            decision_config=root.mnistr.decision,
            snapshotter_config=root.mnistr.snapshotter,
            loader_name=root.mnistr.loader_name,
            loader_config=root.mnistr.loader,
            layers=root.mnistr.layers,
            loss_function=root.mnistr.loss_function)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.snapshotter.time_interval = 0
        workflow.snapshotter.interval = 3 + 1
        workflow.initialize(device=self.device, snapshot=False)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        file_name = workflow.snapshotter.file_name

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 634)
        self.assertEqual(3, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = Snapshotter.import_(file_name)
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 6
        workflow_from_snapshot.decision.complete <<= False
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.initialize(device=self.device, snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        self.assertEqual(err, 474)
        self.assertEqual(6, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
