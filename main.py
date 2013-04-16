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
import os
import evaluator
import argparse
import threading
import gd


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
    if pt.d != "D" or pt.c != "CC" or pt.b != "B" or pt.a != "AA" or pt.h_:
        raise Exception("Pickle test failed.")


def fork_snapshot(obj, file, wait_for_completion = 1):
    """Makes snapshot of obj to the file.

    Wont work with OpenCL buffer mapping during pickle.
    """
    pid = os.fork()
    if pid:
        if wait_for_completion:
            os.waitpid(pid, 0)
        return
    pickle.dump(obj, file)
    file.flush()
    sys.exit()


class EndPoint(filters.Filter):
    """On initialize() and run() releases its semaphore.
    
    Attributes:
        sem_: semaphore.
    """
    def __init__(self, unpickling = 0):
        super(EndPoint, self).__init__(unpickling=unpickling)
        self.sem_ = threading.Semaphore(0)

    def initialize(self):
        self.sem_.release()

    def run(self):
        self.sem_.release()

    def wait(self):
        """Waits on semaphore.
        """
        self.sem_.acquire()


class UseCase1(filters.SmartPickling):
    """Use case 1.

    Attributes:
        device_list: list of an OpenCL devices as DeviceList object.
        start_point: Filter.
        end_point: EndPoint.
        sem_: semaphore.
    """
    def __init__(self, cpu, unpickling = 0):
        super(UseCase1, self).__init__(unpickling=unpickling)
        self.sem_ = threading.Semaphore(0)
        if unpickling:
            return

        dev = None
        if not cpu:
            self.device_list = opencl.DeviceList()
            dev = self.device_list.get_device()

        # Setup notification flow
        m = mnist.MNISTLoader()

        aa = all2all.All2AllTanh(output_shape=[1024], device=dev)
        aa.input = m.output
        aa.link_from(m)

        aa2 = all2all.All2AllTanh(output_shape=[256], device=dev)
        aa2.input = aa.output
        aa2.link_from(aa)

        aa3 = all2all.All2AllTanh(output_shape=[64], device=dev)
        aa3.input = aa2.output
        aa3.link_from(aa2)

        out = all2all.All2AllSoftmax(output_shape=[16], device=dev)
        out.input = aa3.output
        out.link_from(aa3)

        ev = evaluator.BatchEvaluator(device=dev)
        ev.y = out.output
        ev.labels = m.labels
        ev.link_from(out)
        ev.link_from(m)

        gdsm = gd.GDSM(device=dev)
        gdsm.weights = out.weights
        gdsm.bias = out.bias
        gdsm.h = out.input
        gdsm.y = out.output
        gdsm.err_y = ev.err_y
        gdsm.link_from(ev)

        #TODO(a.kazantsev): add other filters

        self.start_point = filters.Filter()
        m.link_from(self.start_point)
        self.end_point = EndPoint()
        self.end_point.link_from(gdsm)

    def run(self, resume = False):
        # Start the process:
        print()
        print("Initializing...")
        self.start_point.initialize_dependent()
        self.end_point.wait()
        print()
        print("Running...")
        self.start_point.run_dependent()
        self.end_point.wait()


def main():
    do_pickle_test()

    # Main program
    logging.debug("Entered")

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action="store_true", help="resume from snapshot", \
                        default=False, dest="resume")
    parser.add_argument("-cpu", action="store_true", help="use numpy only", \
                        default=False, dest="cpu")
    args = parser.parse_args()

    numpy.random.seed(numpy.fromfile("seed", numpy.integer))

    uc = None
    if args.resume:
        try:
            print("Resuming from snapshot...")
            fin = open("cache/snapshot.pickle", "rb")
            (uc, random_state) = pickle.load(fin)
            numpy.random.set_state(random_state)
            fin.close()
        except IOError:
            print("Could not resume from cache/snapshot.pickle")
            uc = None
    if not uc:
        uc = UseCase1(args.cpu)
    print("Launching...")
    uc.run(args.resume)

    print()
    print("Snapshotting...")
    fout = open("cache/snapshot.pickle", "wb")
    #fork_snapshot((uc, numpy.random.get_state()), fout)
    pickle.dump((uc, numpy.random.get_state()), fout)
    fout.close()
    print("Done")

    logging.debug("Finished")
    sys.exit()


if __name__ == '__main__':
    main()
