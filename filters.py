"""
Created on Mar 12, 2013

Filters in data stream neural network model.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import os
import sys
import pickle
import error
import time
import numpy
import _thread
import threading


class SmartPickling(object):
    """Will save attributes ending with _ as None when pickling and will call constructor upon unpickling.
    """
    def __init__(self, unpickling = 0):
        """Constructor.
        
        Parameters:
            unpickling: if 1, object is being created via unpickling.
        """
        pass

    def __getstate__(self):
        """What to pickle.
        """
        state = {}
        for k, v in self.__dict__.items():
            if k[len(k) - 1] != "_":
                state[k] = v
            else:
                state[k] = None
        return state

    def __setstate__(self, state):
        """What to unpickle.
        """
        self.__dict__.update(state)
        self.__init__(unpickling=1)


class State(SmartPickling):
    """State of the filter.

    Attributes:
        mtime: time of the last modification.
    """
    def __init__(self, unpickling = 0):
        super(State, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.mtime = 0.0

    def update_mtime(self):
        """Update mtime that it will become greater than already and trying set it with system time first.
        """
        mtime = time.time()
        if mtime <= self.mtime:
            dt = 0.000001
            mtime = self.mtime + dt
            while mtime <= self.mtime:
                mtime += dt
                dt += dt
        self.mtime = mtime


class Event(SmartPickling):
    """Event object.

    Attributes:
        active: active event or not.
        lock_: lock.
        sem_: semaphore to raise on set.
        ready_queue: list where to append self on set.
        owner: owner of the event.
    """
    def __init__(self, active = 0, unpickling = 0):
        super(Event, self).__init__(unpickling=unpickling)
        self.lock_ = _thread.allocate_lock()
        self.active = active
        self.sem_ = None
        self.ready_queue = None
        self.owner = None

    def set(self):
        """Sets the event to active.
        """
        self.lock_.acquire()
        self.active = 1
        if self.sem_:
            self.ready_queue.append(self)
            self.sem_.release()
        self.lock_.release()

    def attach(self, sem, ready_queue, owner):
        """Attaches object to the event queue.
        """
        self.lock_.acquire()
        self.sem_ = sem
        self.ready_queue = ready_queue
        self.owner = owner
        if self.active and self.sem_:
            self.ready_queue.append(self)
            self.sem_.release()
        self.lock_.release()

    def post_check(self):
        """Called once after event is ready, should raise Exception if there is a error.
        """
        pass


class OpenCLEvent(Event):
    """OpenCL event object.
    
    Attributes:
        ev_: pyopencl.Event.
        arr_: first argument returned in case of pyopencl.enqueue_map_buffer().
    """
    def __init__(self, ev, arr = None, unpickling = 0):
        super(OpenCLEvent, self).__init__(unpickling=unpickling)
        self.ev_ = ev
        self.arr_ = arr

    def attach(self, sem, ready_queue, owner):
        super(OpenCLEvent, self).attach(sem, ready_queue, owner)
        _thread.start_new_thread(self.run, ())

    def run(self):
        self.ev_.wait()
        self.set()

    def post_check(self):
        if self.arr_ != None:
            self.arr_.base.release(queue=self.owner.device.queue_, wait_for=None)
        if self.ev_.command_execution_status < 0:
            raise error.ErrOpenCL(self)


class GeneralFilter(State):
    """General filter in data stream neural network model.
    
    Attributes:
        random_state: numpy random state.
    """
    def __init__(self, unpickling = 0):
        super(GeneralFilter, self).__init__(unpickling=unpickling)
        if unpickling:
            if self.random_state:
                numpy.random.set_state(self.random_state)
            return
        self.random_state = ()

    def snapshot(self, file, wait_for_completion = 1):
        """Makes snapshot to the file.
        """
        pid = os.fork()
        if pid:
            if wait_for_completion:
                os.waitpid(pid, 0)
            return
        self.random_state = numpy.random.get_state()
        pickle.dump(self, file)
        file.flush()
        sys.exit()

    def run(self):
        """Start the job.
        
        Returns:
            Event.
        """
        pass

    def post_run(self):
        """Called once just after the end of job started in run().
        """
        pass


def all_values(stt):
    return all(stt.values())


def any_value(stt):
    return any(stt.values())


def always(stt):
    return 1


def never(stt):
    return 0


class Notifications(GeneralFilter):
    """Network of the notifications between filters.
    
    Attributes:
        src_to: what depends on src.
        dst_from: on what dst depends.
        gates: dictionary of gate functions by dst.
        sem_: semaphore.
        ready_queue: queue of set events.
        pending_events: dictionary of pending events.
    """
    def __init__(self, unpickling = 0):
        super(Notifications, self).__init__(unpickling=unpickling)
        self.pending_events = {}
        if unpickling:
            self.sem_ = threading.Semaphore(len(self.ready_queue))
            return
        self.ready_queue = []
        self.sem_ = threading.Semaphore(0)
        self.src_to = {}
        self.dst_from = {}
        self.gates = {}

    def set_rule(self, dst_filter, src_filters, gate_function = always):
        """Sets filter activation rule.

        Parameters:
            dst_filter: filter on which to set rule.
            src_filters: list of filters it depends on.
            gate_function: function to be called on activation of any of the src_filters
                gate_function(stt), where:
                    stt is the dictionary: filter => state
                        state in {0, 1}, 1 means activated
                it's return value lies in {0, 1, 2}:
                    0 - gate closed,
                    1 - gate open and we should reset src_filters_state,
                    2 - gate open and we should not reset src_filters_state.
        """
        if dst_filter in self.dst_from:
            for src in self.dst_from[dst_filter]:
                del(self.src_to[src][dst_filter])
        src_filters_state = {}
        for src in src_filters:
            src_filters_state[src] = 0
            if src not in self.src_to:
                self.src_to[src] = {}
            self.src_to[src][dst_filter] = 1
        self.dst_from[dst_filter] = src_filters_state
        self.gates[dst_filter] = gate_function

    def notify(self, src_filter):
        """Processes activation of the specified filter.
        """
        if src_filter not in self.src_to:
            return
        for dst in self.src_to[src_filter]:
            self.dst_from[dst][src_filter] = 1
            gate = self.gates[dst]
            state = gate(self.dst_from[dst])
            if state:
                if state == 1:
                    for src in self.dst_from[dst].keys():
                        self.dst_from[dst][src] = 0
                ev = dst.run()
                if ev:
                    ev.attach(self.sem_, self.ready_queue, dst)
                    self.pending_events[ev] = 1

    def run(self):
        """Runs self.
        """
        self.notify(self)
        return None

    def notify_next(self):
        """Wait for the next filter to do its job.

        Raises:
            ErrNotExists, ErrOpenCL.
        """
        if not self.pending_events:
            raise error.ErrNotExists
        self.sem_.acquire()
        ev = self.ready_queue.pop(0)
        del(self.pending_events[ev])
        ev.post_check()
        ev.owner.post_run()
        self.notify(ev.owner)


class OpenCLFilter(GeneralFilter):
    """Filter that operates using OpenCL.
    
    Attributes:
        device: Device object.
    """
    def __init__(self, device = None, unpickling = 0):
        super(OpenCLFilter, self).__init__(unpickling = unpickling)
        if unpickling:
            return
        self.device = device


class Batch(State):
    """Batch.

    Attributes:
        batch: numpy array with first dimension as batch.
        batch_: OpenCL buffer mapped to batch.
    """
    def __init__(self, unpickling = 0):
        super(Batch, self).__init__(unpickling=unpickling)
        self.batch_ = None
        if unpickling:
            return
        self.batch = None


class Vector(State):
    """Vector.
    
    Attributes:
        v: numpy array.
        v_: OpenCL buffer mapped to v.
    """
    def __init__(self, unpickling = 0):
        super(Vector, self).__init__(unpickling=unpickling)
        self.v_ = None
        if unpickling:
            return
        self.v = None


class Labels(State):
    """Labels for batch.

    Attributes:
        n_classes: number of classes.
        v: numpy array sized as a batch with one label per image in range [0, n_classes).
    """
    def __init__(self, unpickling = 0):
        super(Labels, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.n_classes = 0
        self.v = None
