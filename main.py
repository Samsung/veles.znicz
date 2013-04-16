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
        status: has completed attribute.
    """
    def __init__(self, unpickling = 0):
        super(EndPoint, self).__init__(unpickling=unpickling)
        self.sem_ = threading.Semaphore(0)
        if unpickling:
            return
        self.status = None

    def initialize(self):
        self.sem_.release()

    def run(self):
        if not self.status.completed:
            return
        self.sem_.release()
        return 1

    def wait(self):
        """Waits on semaphore.
        """
        self.sem_.acquire()


class Repeater(filters.Filter):
    """Propagates notification if any of the inputs are active.
    """
    def __init__(self, unpickling = 0):
        super(Repeater, self).__init__(unpickling=unpickling)
        if unpickling:
            return

    def gate(self, src):
        """Gate is always open.
        """
        return 1


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
        self.start_point = filters.Filter()

        m = mnist.MNISTLoader()
        m.link_from(self.start_point)

        rpt = Repeater()
        rpt.link_from(m)

        aa1 = all2all.All2AllTanh(output_shape=[64], device=dev)
        aa1.input = m.output
        aa1.link_from(rpt)

        aa2 = all2all.All2AllTanh(output_shape=[32], device=dev)
        aa2.input = aa1.output
        aa2.link_from(aa1)

        out = all2all.All2AllSoftmax(output_shape=[16], device=dev)
        out.input = aa2.output
        out.link_from(aa2)

        ev = evaluator.BatchEvaluator(device=dev)
        ev.y = out.output
        ev.labels = m.labels
        ev.link_from(out)

        self.end_point = EndPoint()
        self.end_point.status = ev.status
        self.end_point.link_from(ev)

        gdsm = gd.GDSM(device=dev)
        gdsm.weights = out.weights
        gdsm.bias = out.bias
        gdsm.h = out.input
        gdsm.y = out.output
        gdsm.err_y = ev.err_y
        gdsm.link_from(self.end_point)

        gd2 = gd.GDTanh(device=dev)
        gd2.weights = aa2.weights
        gd2.bias = aa2.bias
        gd2.h = aa2.input
        gd2.y = aa2.output
        gd2.err_y = gdsm.err_h
        gd2.link_from(gdsm)

        gd1 = gd.GDTanh(device=dev)
        gd1.weights = aa1.weights
        gd1.bias = aa1.bias
        gd1.h = aa1.input
        gd1.y = aa1.output
        gd1.err_y = gd2.err_h
        gd1.link_from(gd2)

        rpt.link_from(gd1)

        #TODO(a.kazantsev): ensure that scheme is working as desired

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
