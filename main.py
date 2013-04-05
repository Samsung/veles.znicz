#!/usr/bin/python3
"""
Created on Mar 11, 2013

Entry point.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import logging
import filters
import sys
import mnist
import all2all
import numpy
import opencl
import pickle
import error
import os
import evaluator


g_pt = 0
class PickleTest(filters.SmartPickling):
    """Pickle test.
    """
    def __init__(self, unpickling = 0, a = "A", b = "B", c = "C"):
        global g_pt
        g_pt += 1
        super(PickleTest, self).__init__(unpickling)
        if unpickling:
            return
        self.a = a
        self.b = b
        self.c = c


def do_pickle_test():
    # Test for correct behavior of filters.SmartPickling
    pt = PickleTest(a = "AA", c = "CC")
    if g_pt != 1:
        raise Exception("Pickle test failed.")
    pt.d = "D"
    pt.h_ = "HH"
    try:
        os.mkdir("cache")
    except OSError:
        pass
    fout = open("cache/test.pickle", "wb")
    pickle.dump(pt, fout)
    fout.close()
    del(pt)
    fin = open("cache/test.pickle", "rb")
    pt = pickle.load(fin)
    fin.close()
    if g_pt != 2:
        raise Exception("Pickle test failed.")
    if pt.d != "D" or pt.c != "CC" or pt.b != "B" or pt.a != "AA":
        raise Exception("Pickle test failed.")
    try:
        print("SHOULD_NOT_SEE(Unpickling attribute): " + pt.h_)
        raise Exception("Pickle test failed.")
    except AttributeError:
        pass


def main():
    do_pickle_test()

    # Main program
    logging.debug("Entered")

    numpy.random.seed(numpy.fromfile("seed", numpy.integer))

    # Setup notification flow
    nn = filters.Notifications()

    nn.cl = opencl.OpenCL()
    m = mnist.MNISTLoader()
    nn.set_rule(m, [nn])

    aa = all2all.All2AllTanh(output_shape=[1024])
    nn.set_rule(aa, [m])
    aa.input = m.output

    aa2 = all2all.All2AllTanh(output_shape=[256])
    nn.set_rule(aa2, [aa])
    aa2.input = aa.output

    aa3 = all2all.All2AllTanh(output_shape=[64])
    nn.set_rule(aa3, [aa2])
    aa3.input = aa2.output

    out = all2all.All2AllSoftmax(output_shape=[16])
    nn.set_rule(out, [aa3])
    out.input = aa3.output

    ev = evaluator.BatchEvaluator()
    nn.set_rule(ev, [out])
    ev.input = out.output
    ev.labels = m.labels

    #TODO(a.kazantsev): add other filters

    # Start the process:
    nn.run()
    print(m.labels.n_classes)

    # Run notifications until job is done
    try:
        while True:
            nn.notify_next()
    except error.ErrNotExists:
        pass

    print()
    print("Snapshotting...")
    fout = open("cache/snapshot.pickle", "wb")
    nn.snapshot(fout)
    fout.close()
    print("Done")

    logging.debug("Finished")
    sys.exit()


if __name__ == '__main__':
    main()
