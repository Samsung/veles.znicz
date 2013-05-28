'''
Created on Apr 19, 2013

@author: Seresov Denis <d.seresov@samsung.com>
'''
import units
import threading 
import pickle
import numpy


class EndPointTasks(units.Unit):
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
    def __init__(self, snapshot_object = None, flog = None, flog_args = None, unpickling = 0):
        super(EndPointTasks, self).__init__(unpickling=unpickling)
        self.sem_ = threading.Semaphore(0)
        self.n_passes_ = 0
        if unpickling:
            return
        self.status = None
        self.n_passes = 0
        self.max_passes = 10000
        self.snapshot_frequency = 100
        self.snapshot_filename = "cache/snapshot.%d.pickle"
        self.snapshot_object = snapshot_object
        self.flog_ = flog
        self.flog_args_ = flog_args

    def initialize(self):
        self.sem_.release()

    def run(self):
        self.sem_.release()
        return 1

    def wait(self):
        """Waits on semaphore.
        """
        self.sem_.acquire()
    #def gate(self, src):
    #    """Gate is always open.
    #    """
    #    return 1


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
    def __init__(self, snapshot_object = None, flog = None, flog_args = None, unpickling = 0):
        super(EndPoint, self).__init__(unpickling=unpickling)
        self.sem_ = threading.Semaphore(0)
        self.n_passes_ = 0
        if unpickling:
            return
        self.status = None
        self.n_passes = 0
        self.max_passes = 10000
        self.snapshot_frequency = 100
        self.snapshot_filename = "cache/snapshot.%d.pickle"
        self.snapshot_object = snapshot_object
        self.flog_ = flog
        self.flog_args_ = flog_args

    def initialize(self):
        self.sem_.release()

    def run(self):
        self.n_passes_ += 1
        self.n_passes += 1
        print("It_s (s %d,t %d)" % (self.n_passes_, self.n_passes))
        if self.n_passes % self.snapshot_frequency == 0:
            fnme = self.snapshot_filename % (self.n_passes, )
            print("Snapshotting to %s" % (fnme, ))
            fout = open(fnme, "wb")
            pickle.dump((self.snapshot_object, numpy.random.get_state()), fout)
            fout.close()
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

