"""
Created on Mar 12, 2013

Filters in data stream neural network model.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import os
import sys
import pickle
import error
import numpy
import time
import threading


class SmartPickling(object):
    """Will not pickle attributes ending with _
    """
    def __init__(self, unpickling = 0):
        """Constructor will have one additional argument.
        """
        pass

    def __getstate__(self):
        """What to pickle.
        """
        state = {}
        for k, v in self.__dict__.items():
            if k[len(k) - 1] != "_":
                state[k] = v
        return state

    def __getnewargs__(self):
        """How to call __init__() after unpickling.
        """
        return (1,)

    def __new__(cls, *args, **kwargs):
        """Had to rewrite 'cause python does not call __init__() when unserializing.
        """
        obj = super(SmartPickling, cls).__new__(cls)
        if args and args[0] and obj:
            obj.__init__(*args, **kwargs)
        return obj


class State(SmartPickling):
    """State of the filter.

    Attributes:
        mtime: time of the last modification.
        data: any data.
    """
    def __init__(self, unpickling = 0):
        super(State, self).__init__(unpickling)
        if unpickling:
            return
        self.mtime = 0.0
        self.data = None

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


class GeneralFilter(State):
    """General filter in data stream neural network model.

    Attributes:
        random_state: state of the numpy random.
    """
    def __init__(self, unpickling = 0):
        super(GeneralFilter, self).__init__(unpickling)
        if unpickling:
            if self.random_state:
                numpy.random.set_state(self.random_state)
            return
        self.random_state = ()

    def snapshot(self, file, wait_for_completion = 1, save_random_state = 1):
        """Makes snapshot to the file.
        """
        pid = os.fork()
        if pid:
            if wait_for_completion:
                os.waitpid(pid, 0)
            return
        if save_random_state:
            self.random_state = numpy.random.get_state()
        pickle.dump(self, file)
        sys.exit()

    def run(self, endofjob_callback = None):
        """Do the job.
        
        Parameters:
            endofjob_callback: function that should be called when the job will be done.
        """
        if endofjob_callback:
            endofjob_callback(self)


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
        _sem: semaphore.
        executing: dictionary of the currently executing filters.
        executed: list of filters that has already executed.
        src_to: what depends on src.
        dst_from: on what dst depends.
        gates: dictionary of gate functions by dst.
    """
    def __init__(self, unpickling = 0):
        super(Notifications, self).__init__(unpickling)
        if unpickling:
            self.sem_ = threading.Semaphore(len(self.executed))
            return
        self.sem_ = threading.Semaphore(0)
        self.executing = {}
        self.executed = []
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
                self.executing[dst] = 1
                dst.run(self.endofjob_callback)

    def run(self, endofjob_callback = None):
        """Runs self.
        """
        self.notify(self)

    def endofjob_callback(self, src_filter):
        """Called when the src_filter ends its execution.
        """
        self.executed.append(src_filter)
        del(self.executing[src_filter])
        self.sem_.release()

    def notify_next(self):
        """Wait for the next filter to do its job.

        Raises:
            ErrNotExists.
        """
        if self.executing:
            self.sem_.acquire()
        if not self.executed:
            raise error.ErrNotExists()
        src = self.executed.pop(0)  # FIFO
        self.notify(src)
