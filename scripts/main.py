#!/usr/bin/python3.3
"""
Created on Mar 11, 2013

Entry point.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import logging
import units
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
import rnd
import plotters


class EndPoint(units.Unit):
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
    def __init__(self, snapshot_object=None, flog=None, flog_args=None,
                 unpickling=0):
        super(EndPoint, self).__init__(unpickling=unpickling)
        self.sem_ = threading.Semaphore(0)
        self.n_passes_ = 0
        self.max_passes = 500000
        if unpickling:
            return
        self.status = None
        self.n_passes = 0
        self.snapshot_frequency = 500
        self.snapshot_filename = "cache/snapshot.%d.pickle"
        self.snapshot_object = snapshot_object
        self.flog_ = flog
        self.flog_args_ = flog_args

    def initialize(self):
        self.sem_.release()

    def run(self):
        self.n_passes_ += 1
        self.n_passes += 1
        self.log().info("Iterations (session, total): (%d, %d)\n" % (self.n_passes_,
                                                           self.n_passes))
        if self.n_passes % self.snapshot_frequency == 0:
            fnme = self.snapshot_filename % (self.n_passes,)
            self.log().info("Snapshotting to %s" % (fnme,))
            fout = open(fnme, "wb")
            pickle.dump((self.snapshot_object, numpy.random.get_state()), fout)
            fout.close()
        if self.n_passes_ >= 1000 and \
           self.__dict__.get("max_ok", 0) < self.status.n_ok:
            self.max_ok = self.status.n_ok
            self.log().info("Snapshotting to /tmp/snapshot.best")
            fout = open("/tmp/snapshot.best.tmp", "wb")
            pickle.dump((self.snapshot_object, numpy.random.get_state()), fout)
            fout.close()
            try:
                os.unlink("/tmp/snapshot.best.old")
                os.rename("/tmp/snapshot.best", "/tmp/snapshot.best.old")
            except OSError:
                pass
            os.rename("/tmp/snapshot.best.tmp", "/tmp/snapshot.best")
        if self.flog_:
            self.flog_(*self.flog_args_)
        if self.n_passes_ < self.max_passes and not self.status.completed:
            return
        self.sem_.release()
        return 1

    def wait(self):
        """Waits on semaphore.
        """
        self.sem_.acquire()


class Repeater(units.Unit):
    """Propagates notification if any of the inputs are active.
    """
    def __init__(self, unpickling=0):
        super(Repeater, self).__init__(unpickling=unpickling)
        if unpickling:
            return

    def gate(self, src):
        """Gate is always open.
        """
        return 1


class UseCase1(units.SmartPickler):
    """MNIST with softmax.

    Attributes:
        device_list: list of an OpenCL devices as DeviceList object.
        start_point: Unit.
        end_point: EndPoint.
        aa1: aa1.
        aa2: aa2.
        sm: softmax.
        ev: evaluator.
        gdsm: gdsm.
        gd2: gd2.
        gd1: gd1.
    """
    def __init__(self, cpu=False, unpickling=0):
        super(UseCase1, self).__init__(unpickling=unpickling)
        if unpickling:
            return

        dev = None
        if not cpu:
            self.device_list = opencl.DeviceList()
            dev = self.device_list.get_device()

        # Setup notification flow
        self.start_point = units.Unit()

        m = mnist.MNISTLoader()
        m.link_from(self.start_point)

        rpt = Repeater()
        rpt.link_from(m)

        aa1 = all2all.All2AllTanh(output_shape=[120], device=dev)
        aa1.input = m.output
        aa1.link_from(rpt)

        aa2 = all2all.All2AllTanh(output_shape=[120], device=dev)
        aa2.input = aa1.output
        aa2.link_from(aa1)

        aa3 = all2all.All2AllTanh(output_shape=[120], device=dev)
        aa3.input = aa2.output
        aa3.link_from(aa2)

        aa4 = all2all.All2AllTanh(output_shape=[120], device=dev)
        aa4.input = aa3.output
        aa4.link_from(aa3)

        aa5 = all2all.All2AllTanh(output_shape=[120], device=dev)
        aa5.input = aa4.output
        aa5.link_from(aa4)

        aa6 = all2all.All2AllTanh(output_shape=[120], device=dev)
        aa6.input = aa5.output
        aa6.link_from(aa5)

        aa7 = all2all.All2AllTanh(output_shape=[120], device=dev)
        aa7.input = aa6.output
        aa7.link_from(aa6)

        sm = all2all.All2AllSoftmax(output_shape=[10], device=dev)
        sm.input = aa7.output
        sm.link_from(aa7)

        ev = evaluator.EvaluatorSoftmax(device=dev)
        ev.y = sm.output
        ev.labels = m.labels
        ev.link_from(sm)

        plt = plotters.SimplePlotter(device=dev,
                                     figure_label="num errors")
        plt.input = ev.status
        plt.input_field = 'num_errors'
        plt.link_from(ev)

        self.end_point = EndPoint(self)
        self.end_point.status = ev.status
        self.end_point.link_from(ev)

        gdsm = gd.GDASM(device=dev)
        gdsm.weights = sm.weights
        gdsm.bias = sm.bias
        gdsm.h = sm.input
        gdsm.y = sm.output
        gdsm.err_y = ev.err_y
        gdsm.link_from(self.end_point)

        gd7 = gd.GDATanh(device=dev)
        gd7.weights = aa7.weights
        gd7.bias = aa7.bias
        gd7.h = aa7.input
        gd7.y = aa7.output
        gd7.err_y = gdsm.err_h
        gd7.link_from(gdsm)

        gd6 = gd.GDATanh(device=dev)
        gd6.weights = aa6.weights
        gd6.bias = aa6.bias
        gd6.h = aa6.input
        gd6.y = aa6.output
        gd6.err_y = gd7.err_h
        gd6.link_from(gd7)

        gd5 = gd.GDATanh(device=dev)
        gd5.weights = aa5.weights
        gd5.bias = aa5.bias
        gd5.h = aa5.input
        gd5.y = aa5.output
        gd5.err_y = gd6.err_h
        gd5.link_from(gd6)

        gd4 = gd.GDATanh(device=dev)
        gd4.weights = aa4.weights
        gd4.bias = aa4.bias
        gd4.h = aa4.input
        gd4.y = aa4.output
        gd4.err_y = gd5.err_h
        gd4.link_from(gd5)

        gd3 = gd.GDATanh(device=dev)
        gd3.weights = aa3.weights
        gd3.bias = aa3.bias
        gd3.h = aa3.input
        gd3.y = aa3.output
        gd3.err_y = gd4.err_h
        gd3.link_from(gd4)

        gd2 = gd.GDATanh(device=dev)
        gd2.weights = aa2.weights
        gd2.bias = aa2.bias
        gd2.h = aa2.input
        gd2.y = aa2.output
        gd2.err_y = gd3.err_h
        gd2.link_from(gd3)

        gd1 = gd.GDATanh(device=dev)
        gd1.weights = aa1.weights
        gd1.bias = aa1.bias
        gd1.h = aa1.input
        gd1.y = aa1.output
        gd1.err_y = gd2.err_h
        gd1.link_from(gd2)

        rpt.link_from(gd1)

        self.m = m
        self.aa1 = aa1
        self.aa2 = aa2
        self.aa3 = aa3
        self.aa4 = aa4
        self.aa5 = aa5
        self.aa6 = aa6
        self.aa7 = aa7
        self.sm = sm
        self.ev = ev
        self.gdsm = gdsm
        self.gd7 = gd7
        self.gd6 = gd6
        self.gd5 = gd5
        self.gd4 = gd4
        self.gd3 = gd3
        self.gd2 = gd2
        self.gd1 = gd1

    def run(self, resume=False, global_alpha=0.9, global_lambda=0.0,
            threshold=1.0, threshold_low=None, test_only=False):
        # Start the process:
        self.m.test_only = test_only
        self.ev.threshold = threshold
        self.ev.threshold_low = threshold_low
        self.gdsm.global_alpha = global_alpha
        self.gdsm.global_lambda = global_lambda
        self.gd7.global_alpha = global_alpha
        self.gd7.global_lambda = global_lambda
        self.gd6.global_alpha = global_alpha
        self.gd6.global_lambda = global_lambda
        self.gd5.global_alpha = global_alpha
        self.gd5.global_lambda = global_lambda
        self.gd4.global_alpha = global_alpha
        self.gd4.global_lambda = global_lambda
        self.gd3.global_alpha = global_alpha
        self.gd3.global_lambda = global_lambda
        self.gd2.global_alpha = global_alpha
        self.gd2.global_lambda = global_lambda
        self.gd1.global_alpha = global_alpha
        self.gd1.global_lambda = global_lambda
        self.ev.origin = self.aa1.input
        if test_only:
            self.end_point.max_passes = 1
        self.log().info()
        self.log().info("Initializing...")
        self.start_point.initialize_dependent()
        self.end_point.wait()
        self.log().info()
        self.log().info("Running...")
        self.start_point.run_dependent()
        self.end_point.wait()


class UseCase2(units.SmartPickler):
    """Wine with Softmax.

    Attributes:
        device_list: list of an OpenCL devices as DeviceList object.
        start_point: Unit.
        end_point: EndPoint.
        t: t.
    """
    def __init__(self, cpu=True, unpickling=0):
        super(UseCase2, self).__init__(unpickling=unpickling)
        if unpickling:
            return

        dev = None
        if not cpu:
            self.device_list = opencl.DeviceList()
            dev = self.device_list.get_device()

        # Setup notification flow
        self.start_point = units.Unit()

        # m = mnist.MNISTLoader()
        t = text.TXTLoader()
        self.t = t
        # sys.exit()
        self.log().debug("1")
        t.link_from(self.start_point)
        self.log().debug("2")

        rpt = Repeater()
        rpt.link_from(t)

        aa1 = all2all.All2AllTanh(output_shape=[5], device=dev)
        aa1.input = t.output2
        aa1.link_from(rpt)

        out = all2all.All2AllSoftmax(output_shape=[3], device=dev)
        out.input = aa1.output
        out.link_from(aa1)

        ev = evaluator.EvaluatorSoftmax(device=dev)
        ev.y = out.output
        ev.labels = t.labels
        ev.link_from(out)

        plt = plotters.SimplePlotter(device=dev,
                                     figure_label="num errors")
        plt.input = ev.status
        plt.input_field = 'num_errors'
        plt.link_from(ev)

        gdsm = gd.GDSM(device=dev)
        gdsm.weights = out.weights
        gdsm.bias = out.bias
        gdsm.h = out.input
        gdsm.y = out.output
        gdsm.err_y = ev.err_y

        gd1 = gd.GDTanh(device=dev)
        gd1.weights = aa1.weights
        gd1.bias = aa1.bias
        gd1.h = aa1.input
        gd1.y = aa1.output
        gd1.err_y = gdsm.err_h
        gd1.link_from(gdsm)

        rpt.link_from(gd1)

        self.end_point = EndPoint(self)
        self.end_point.status = ev.status
        self.end_point.link_from(ev)
        gdsm.link_from(self.end_point)

        self.sm = out
        self.gdsm = gdsm
        self.gd1 = gd1

    def run(self, resume=False, global_alpha=0.9, global_lambda=0.0,
            threshold=1.0, threshold_low=None, test_only=False):
        # Start the process:
        self.sm.threshold = threshold
        self.sm.threshold_low = threshold_low
        self.gdsm.global_alpha = global_alpha
        self.gdsm.global_lambda = global_lambda
        self.gd1.global_alpha = global_alpha
        self.gd1.global_lambda = global_lambda
        self.log().info()
        self.log().info("Initializing...")
        self.start_point.initialize_dependent()
        self.end_point.wait()
        self.log().info()
        self.log().info("Running...")
        self.start_point.run_dependent()
        self.end_point.wait()


class UseCase3(units.SmartPickler):
    """Wine with MSE.

    Attributes:
        device_list: list of an OpenCL devices as DeviceList object.
        start_point: Unit.
        end_point: EndPoint.
        t: t.
    """
    def __init__(self, cpu=True, unpickling=0):
        super(UseCase3, self).__init__(unpickling=unpickling)
        if unpickling:
            return

        dev = None
        if not cpu:
            self.device_list = opencl.DeviceList()
            dev = self.device_list.get_device()

        # Setup notification flow
        self.start_point = units.Unit()

        # m = mnist.MNISTLoader()
        t = text.TXTLoader()
        self.t = t
        # sys.exit()
        self.log().debug("1")
        t.link_from(self.start_point)
        self.log().debug("2")

        rpt = Repeater()
        rpt.link_from(t)

        aa1 = all2all.All2AllTanh(output_shape=[5], device=dev)
        aa1.input = t.output2
        aa1.link_from(rpt)

        out = all2all.All2AllTanh(output_shape=[3], device=dev)
        out.input = aa1.output
        out.link_from(aa1)

        ev = evaluator.EvaluatorMSE(device=dev)
        ev.y = out.output
        ev.labels = t.labels
        ev.link_from(out)

        plt = plotters.SimplePlotter(device=dev,
                                     figure_label="num errors")
        plt.input = ev.status
        plt.input_field = 'num_errors'
        plt.link_from(ev)

        gd0 = gd.GDTanh(device=dev)
        gd0.weights = out.weights
        gd0.bias = out.bias
        gd0.h = out.input
        gd0.y = out.output
        gd0.err_y = ev.err_y

        gd1 = gd.GDTanh(device=dev)
        gd1.weights = aa1.weights
        gd1.bias = aa1.bias
        gd1.h = aa1.input
        gd1.y = aa1.output
        gd1.err_y = gd0.err_h
        gd1.link_from(gd0)

        rpt.link_from(gd1)

        self.end_point = EndPoint(self)
        self.end_point.status = ev.status
        self.end_point.link_from(ev)
        gd0.link_from(self.end_point)

        self.sm = out
        self.gd0 = gd0
        self.gd1 = gd1

    def run(self, resume=False, global_alpha=0.9, global_lambda=0.0,
            threshold=1.0, threshold_low=None, test_only=False):
        # Start the process:
        self.sm.threshold = threshold
        self.sm.threshold_low = threshold_low
        self.gd0.global_alpha = global_alpha
        self.gd0.global_lambda = global_lambda
        self.gd1.global_alpha = global_alpha
        self.gd1.global_lambda = global_lambda
        self.log().info()
        self.log().info("Initializing...")
        self.start_point.initialize_dependent()
        self.end_point.wait()
        self.log().info()
        self.log().info("Running...")
        self.start_point.run_dependent()
        self.end_point.wait()


def main():
    # Main program
    logging.debug("Entered")

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, help="resume from snapshot",
                        default="", dest="resume")
    parser.add_argument("-cpu", action="store_true", help="use numpy only",
                        default=False, dest="cpu")
    parser.add_argument("-global_alpha", type=float,
                        help="global gradient descent speed",
                        default=0.9, dest="global_alpha")
    parser.add_argument("-global_lambda", type=float,
                        help="global weights regularisation constant",
                        default=0.0, dest="global_lambda")
    parser.add_argument("-threshold", type=float, help="softmax threshold",
                        default=1.0, dest="threshold")
    parser.add_argument("-threshold_low", type=float,
                        help="softmax threshold low bound",
                        default=None, dest="threshold_low")
    parser.add_argument("-t", action="store_true", help="test only",
                        default=False, dest="test_only")
    args = parser.parse_args()

    rnd.default.seed(numpy.fromfile("seed", numpy.integer, 1024))
    # state = numpy.random.get_state()
    # numpy.random.seed(numpy.fromfile("/dev/urandom", numpy.integer, 1024))
    # numpy.random.set_state(state)

    os.chdir("..")

    uc = None
    if args.resume:
        try:
            logging.info("Resuming from snapshot...")
            fin = open(args.resume, "rb")
            (uc, random_state) = pickle.load(fin)
            numpy.random.set_state(random_state)
            fin.close()
        except IOError:
            logging.error("Could not resume from %s" % (args.resume,))
            uc = None
    if not uc:
        uc = UseCase1(args.cpu)
        # uc = UseCase2(args.cpu)
        # uc = UseCase3(args.cpu)
    logging.info("Launching...")
    uc.run(args.resume, global_alpha=args.global_alpha,
           global_lambda=args.global_lambda, threshold=args.threshold,
           threshold_low=args.threshold_low, test_only=args.test_only)

    logging.info()
    if not args.test_only:
        logging.info("Snapshotting...")
        fout = open("cache/snapshot.pickle", "wb")
        pickle.dump((uc, numpy.random.get_state()), fout)
        fout.close()
        logging.info("Done")

    plotters.Graphics().wait_finish()
    logging.debug("Finished")


if __name__ == '__main__':
    main()
