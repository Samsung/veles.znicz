"""
Created on Mar 13, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


from workflows import Workflow


class DummyLauncher(object):
    @property
    def is_slave(self):
        return False

    @property
    def is_master(self):
        return False

    @property
    def is_standalone(self):
        return True


class DummyWorkflow(Workflow):
    """
    Dummy standalone workflow for unit tests.
    """

    def __init__(self):
        """
        Sets self._launcher_ to DummyLauncher
        """
        super(DummyWorkflow, self).__init__()
        self.launcher = DummyLauncher()
