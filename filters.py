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
        obj = super(SmartPickling, cls).__new__(cls, *args, **kwargs)
        if args and args[0] and obj:
            obj.__init__(*args, **kwargs)
        return obj


class OutputData(SmartPickling):
    """Output data
    
    Attributes:
        mtime: timestamp of the last modification
    """
    def __init__(self, unpickling = 0):
        super(OutputData, self).__init__(unpickling)
        if unpickling:
            return
        self.mtime = 0.0

    def update_mtime(self):
        mtime = time.time()
        if mtime <= self.mtime:
            dt = 0.000001
            mtime = self.mtime + dt
            while mtime <= self.mtime:
                mtime += dt
                dt += dt
        self.mtime = mtime


class GeneralFilter(SmartPickling):
    """General filter in data stream neural network model.

    Attributes:
        parent: parent filter for output_changed() notification.
        output: OutputData of the filter.
        random_state: state of the numpy random.
    """
    def __init__(self, unpickling = 0, parent = None):
        super(GeneralFilter, self).__init__(unpickling)
        if unpickling:
            if self.random_state:
                numpy.random.set_state(self.random_state)
            return
        self.parent = parent
        self.output = None
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

    def input_changed(self, src):
        """Callback, fired when output data of the src, connected to current filter, changes.
        """
        pass

    def output_changed(self, src):
        """Callback, fired on parent when output data of the src changes.
        
        input_changed() should be called on all filters, connected to src.
        """
        pass


class ContainerFilter(GeneralFilter):
    """Filter that contains other filters.

    Attributes:
        filters: dictionary of filters in the container.
        links: dictionary of links between filters.
    """
    def __init__(self, unpickling = 0):
        super(ContainerFilter, self).__init__(unpickling)
        if unpickling:
            return
        self.filters = {}
        self.links = {}

    def add(self, flt):
        """Adds filter to the container.

        Args:
            flt: filter to add.

        Returns:
            flt.

        Raises:
            ErrExists.
        """
        if flt in self.filters:
            raise error.ErrExists()
        self.filters[flt] = 1
        return flt

    def link(self, src, dst):
        """Links to filters

        Args:
            src: source filter
            dst: destination filter

        Returns:
            dst.

        Raises:
            ErrNotExists
            ErrExists
        """
        if (src not in self.filters) or (dst not in self.filters):
            raise error.ErrNotExists()
        if src not in self.links:
            self.links[src] = {}
        if dst in self.links[src]:
            raise error.ErrExists()
        self.links[src][dst] = 1
        return dst

    def output_changed(self, src):
        """GeneralFilter method.
        """
        if src not in self.filters:
            raise error.ErrNotExists()
        if src not in self.links:
            return
        for dst in self.links[src].keys():
            dst.input_changed(src)
