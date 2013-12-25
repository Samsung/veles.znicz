#!/usr/bin/python3.3 -O
"""
Created on Dec 23, 2013

MNIST via DBN.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import logging
import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s" % (this_dir))
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import numpy
import rnd
import opencl
import plotters
import mnist
import rbm
import all2all
import evaluator
import gd
import decision
import workflow
import config


class Loader(mnist.Loader):
    """Loads MNIST dataset.
    """
    def load_data(self):
        """Here we will load MNIST data.
        """
        super(Loader, self).load_data()

        self.class_target.reset()
        self.class_target.v = numpy.zeros([10, 10],
            dtype=config.dtypes[config.dtype])
        for i in range(10):
            self.class_target.v[i, :] = -1
            self.class_target.v[i, i] = 1
        self.original_target = numpy.zeros([self.original_labels.shape[0],
                                            self.class_target.v.shape[1]],
            dtype=self.original_data.dtype)
        for i in range(0, self.original_labels.shape[0]):
            label = self.original_labels[i]
            self.original_target[i] = self.class_target.v[label]

        # At the beginning, initialize values to be found with zeros.
        # NN should be trained in the same way as it will be tested.
        v = self.original_data
        v = v.reshape(v.shape[0], v.size // v.shape[0])
        self.original_data = numpy.zeros([v.shape[0],
            v.shape[1] + self.class_target.v.shape[1]], dtype=v.dtype)
        self.original_data[:, :v.shape[1]] = v[:, :]


class Workflow(workflow.NNWorkflow):
    """Sample workflow.
    """
    def __init__(self, layers, recursion_depth, device):
        super(Workflow, self).__init__(device=device)
        self.rpt.link_from(self.start_point)

        self.loader = Loader()
        self.loader.link_from(self.rpt)

        self.recursion_depth = recursion_depth
        for i in range(recursion_depth):
            aa = rbm.RBMTanh(output_shape=[layers[-1]], device=device)
            self.forward.append(aa)
            if i:
                self.forward[-1].link_from(self.forward[-2])
                self.forward[-1].input = self.forward[-2].output
                self.forward[-1].weights = self.forward[-3].weights
                self.forward[-1].bias = self.forward[-3].bias
            else:
                self.forward[-1].link_from(self.loader)
                self.forward[-1].input = self.loader.minibatch_data

            aa = all2all.All2AllTanh(output_shape=self.forward[-1].input,
                                     device=device, weights_transposed=True)
            self.forward.append(aa)
            self.forward[-1].link_from(self.forward[-2])
            self.forward[-1].input = self.forward[-2].output
            self.forward[-1].weights = self.forward[-2].weights
            if i:
                self.forward[-1].weights = self.forward[-3].weights
                self.forward[-1].bias = self.forward[-3].bias

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorMSE(device=device)
        self.ev.link_from(self.forward[-1])
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.target = self.forward[0].output
        self.ev.max_samples_per_epoch = self.loader.total_samples
        self.ev.class_target = self.loader.class_target

        # Add decision unit
        self.decision = decision.Decision(snapshot_prefix="mnist_dbn")
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_metrics = self.ev.metrics
        self.decision.minibatch_n_err = self.ev.n_err
        self.decision.class_samples = self.loader.class_samples
        self.decision.workflow = self

        # Add gradient descent units
        self.gd.clear()
        self.gd.extend(None for i in range(len(self.forward)))
        self.gd[-1] = gd.GDTanh(device=device, weights_transposed=True)
        self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(device=device,
                                   weights_transposed=((i & 1) != 0))
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
        self.rpt.link_from(self.gd[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = self.decision.complete
        self.end_point.gate_block_not = [1]

        self.loader.gate_block = self.decision.complete

    def initialize(self, device, args):
        super(Workflow, self).initialize(device=device)
        return self.start_point.initialize_dependent()


import argparse
import pickle
import re


def main():
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, default="",
        help="Snapshot with trained network (default empty)")
    parser.add_argument("-snapshot_prefix", type=str, required=True,
        help="Snapshot prefix (Ex.: mnist_dbn_2000)")
    parser.add_argument("-layers", type=str, required=True,
        help="NN layer sizes, separated by any separator (Ex.: 2000)")
    parser.add_argument("-minibatch_size", type=int,
        help="Minibatch size (default 108)", default=108)
    parser.add_argument("-recursion_depth", type=int,
        help="Depth of the RBM's recursive pass (default 1)", default=1)
    parser.add_argument("-global_alpha", type=float,
        help="Global Alpha (default 0.01)", default=0.01)
    parser.add_argument("-global_lambda", type=float,
        help="Global Lambda (default 0.00005)", default=0.00005)
    args = parser.parse_args()

    s_layers = re.split("\D+", args.layers)
    layers = []
    for s in s_layers:
        layers.append(int(s))
    logging.info("Will train NN with layers: %s" % (" ".join(
                                        str(x) for x in layers)))

    global this_dir
    rnd.default.seed(numpy.fromfile("%s/seed" % (this_dir),
                                    numpy.int32, 1024))
    rnd.default2.seed(numpy.fromfile("%s/seed2" % (this_dir),
                                    numpy.int32, 1024))
    device = opencl.Device()
    try:
        fin = open(args.snapshot, "rb")
        w = pickle.load(fin)
        fin.close()
    except IOError:
        w = Workflow(layers=layers, recursion_depth=args.recursion_depth,
                     device=device)
    w.initialize(device=device, args=args)
    w.run()

    plotters.Graphics().wait_finish()
    logging.debug("End of job")


if __name__ == "__main__":
    main()
    sys.exit(0)
