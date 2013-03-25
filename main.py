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


g_pt = 0
class PickleTest(filters.SmartPickling):
    """
    """
    def __init__(self, unpickling = 0, a = "A", b = "B", c = "C"):
        global g_pt
        g_pt += 1
        super(PickleTest, self).__init__(unpickling)
        self.a = a
        self.b = b
        self.c = c


def main():
    # Test for correct behavior of filters.SmartPickling
    pt = PickleTest(a = "AA", c = "CC")
    if g_pt != 1:
        raise Exception("Pickle test failed.")
    pt.d = "D"
    fout = open("cache/test", "wb")
    pickle.dump(pt, fout)
    fout.close()
    del(pt)
    fin = open("cache/test", "rb")
    pt = pickle.load(fin)
    fin.close()
    if g_pt != 2:
        raise Exception("Pickle test failed.")
    if pt.d != "D" or pt.c != "CC" or pt.b != "B" or pt.a != "AA":
        raise Exception("Pickle test failed.")
    del(pt)

    # Main program
    logging.debug("Entered")

    numpy.random.seed(numpy.fromfile("seed", numpy.integer))

    c = filters.ContainerFilter()
    c.cl = opencl.OpenCL()
    m = mnist.MNISTLoader(parent=c)
    c.add(m)

    aa = all2all.All2AllTanh(parent=c, output_layer_size=1024)
    c.add(aa)
    c.link(m, aa)

    #TODO(a.kazantsev): add other filters

    # Start the process:
    m.input_changed(None)

    print()
    print("Snapshotting...")
    fout = open("cache/snapshot.pickle", "wb")
    c.snapshot(fout)
    fout.close()
    print("Done")

    logging.debug("Finished")
    sys.exit()


if __name__ == '__main__':
    main()
