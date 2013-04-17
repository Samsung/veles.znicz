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
import text

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
        n_passes: number of passes.
        n_passes_: number of passes in this session.
        max_passes: maximum number of passes per session before stop.
        snapshot_frequency: frequency of snapshots in number of passes.
        snapshot_object: object to snapshot.
        snapshot_filename: filename with optional %d as snapshot number.
    """
    def __init__(self, snapshot_object = None, unpickling = 0):
        super(EndPoint, self).__init__(unpickling=unpickling)
        self.sem_ = threading.Semaphore(0)
        self.n_passes_ = 0
        if unpickling:
            return
        self.status = None
        self.n_passes = 0
        self.max_passes = 1000
        self.snapshot_frequency = 100
        self.snapshot_filename = "cache/snapshot.%d.pickle"
        self.snapshot_object = snapshot_object

    def initialize(self):
        self.sem_.release()

    def run(self):
        self.n_passes_ += 1
        self.n_passes += 1
        print("Iterations (session, total): (%d, %d)\n" % (self.n_passes_, self.n_passes))
        if self.n_passes % self.snapshot_frequency == 0:
            fnme = self.snapshot_filename % (self.n_passes, )
            print("Snapshotting to %s" % (fnme, ))
            fout = open(fnme, "wb")
            pickle.dump((self.snapshot_object, numpy.random.get_state()), fout)
            fout.close()
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


class UseCase2(filters.SmartPickling):
    """Use case 2.

    Attributes:
        device_list: list of an OpenCL devices as DeviceList object.
        start_point: Filter.
        end_point: EndPoint.
        sem_: semaphore.
        t: t.
    """
    def __init__(self, cpu = True, unpickling = 0):
        super(UseCase2, self).__init__(unpickling=unpickling)
        self.sem_ = threading.Semaphore(0)
        if unpickling:
            return

        dev = None
        if not cpu:
            self.device_list = opencl.DeviceList()
            dev = self.device_list.get_device()

        # Setup notification flow
        self.start_point = filters.Filter()

        #m = mnist.MNISTLoader()
        t =text.TXTLoader()
        self.t = t
        #sys.exit()
        print("1")
        t.link_from(self.start_point)
        print("2")

        rpt = Repeater()
        rpt.link_from(t)

        aa1 = all2all.All2AllTanh(output_shape=[10], device=dev)
        aa1.input = t.output2
        aa1.link_from(rpt)

        out = all2all.All2AllSoftmax(output_shape=[3], device=dev)
        out.input = aa1.output
        out.link_from(aa1)

        ev = evaluator.BatchEvaluator(device=dev)
        ev.y = out.output
        ev.labels = t.labels
        ev.link_from(out)

        self.end_point = EndPoint(self)
        self.end_point.status = ev.status
        self.end_point.link_from(ev)

        gdsm = gd.GDSM(device=dev)
        gdsm.weights = out.weights
        gdsm.bias = out.bias
        gdsm.h = out.input
        gdsm.y = out.output
        gdsm.err_y = ev.err_y
        gdsm.link_from(self.end_point)

        gd1 = gd.GDTanh(device=dev)
        gd1.weights = aa1.weights
        gd1.bias = aa1.bias
        gd1.h = aa1.input
        gd1.y = aa1.output
        gd1.err_y = gdsm.err_h
        gd1.link_from(gdsm)

        rpt.link_from(gd1)
        print("3")

        #TODO(a.kazantsev): ensure that scheme is working as desired

    def run(self, resume = False):
        # Start the process:
        print()
        print("Initializing...")
        self.start_point.initialize_dependent()
        self.end_point.wait()
        #for l in self.t.labels.batch:
        #    print(l)
        #sys.exit()
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
                        default=True, dest="cpu")
    args = parser.parse_args()

    numpy.random.seed(numpy.fromfile("seed", numpy.integer))

    uc = None
    if args.resume:
        try:
            print("Resuming from snapshot...")
            fin = open("cache/uc2.pickle", "rb")
            (uc, random_state) = pickle.load(fin)
            numpy.random.set_state(random_state)
            fin.close()
        except IOError:
            print("Could not resume from cache/snapshot.pickle")
            uc = None
    if not uc:
        uc = UseCase2(args.cpu)
    print("Launching...")
    uc.run(args.resume)

    print()
    print("Snapshotting...")
    fout = open("cache/uc2.pickle", "wb")
    #fork_snapshot((uc, numpy.random.get_state()), fout)
    pickle.dump((uc, numpy.random.get_state()), fout)
    fout.close()
    print("Done")

    logging.debug("Finished")
    sys.exit()


if __name__ == '__main__':
    main()
