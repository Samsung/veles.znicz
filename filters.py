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


class OutputData(object):
    """Output data
    
    Attributes:
        mtime: timestamp of the last modification (some integer)
    """
    def __init__(self):
        super(OutputData, self).__init__()
        self.mtime = 0


class GeneralFilter(object):
    """General filter in data stream neural network model.

    Attributes:
        output: OutputData of the filter.
        parent: parent filter for output_changed() notification.
        random_state: state of the numpy random
    """
    def __init__(self, parent = None):
        super(GeneralFilter, self).__init__()
        self.output = OutputData()
        self.parent = parent
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

    def restore(self):
        """Initialize an object after restoring from snapshot
        """
        if self.random_state:
            numpy.random.set_state(self.random_state)

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
    def __init__(self):
        super(ContainerFilter, self).__init__()
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
