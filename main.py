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


def main():
    logging.debug("Entered")

    numpy.random.seed(numpy.fromfile("seed", numpy.integer))

    c = filters.ContainerFilter()
    c.cl = opencl.OpenCL() 
    m = mnist.MNISTLoader(c)
    c.add(m)

    aa = all2all.All2All(c, 1024)
    c.add(aa)
    c.link(m, aa)

    #TODO(a.kazantsev): add other filters

    # Start the process:
    m.input_changed(None)

    print()
    print("Snapshotting...")
    fout = open("snapshot.pickle", "wb")
    c.snapshot(fout)
    fout.close()
    print("Done")

    logging.debug("Finished")
    sys.exit()


if __name__ == '__main__':
    main()
