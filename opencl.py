"""
Created on Mar 21, 2013

OpenCL helper class.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import pyopencl


class DeviceInfo(object):
    """Some useful preprocessed OpenCL device info.
    
    Attributes:
        memsize: total memory size available for the program.
        rating: relative performance rating of the device.
    """
    def __init__(self, device):
        super(DeviceInfo, self).__init__()
        # MAX_MEM_ALLOC_SIZE usually incorrectly returns only 25% of  the device RAM
        # We will return slightly less amount than the total device RAM
        self.memsize = device.get_info(pyopencl.device_info.GLOBAL_MEM_SIZE) * 9 // 10;
        self.rating = 1.0


class OpenCL(object):
    """OpenCL helper class.

    TODO(a.kazantsev): restrict devices available.

    Attributes:
        platforms: list of OpenCL platforms.
        devices: dictionary of devices per platform
        infos: dictionary of infos per device
        contexts: dictionary of contexts per platform
    """
    def __init__(self):
        super(OpenCL, self).__init__()
        self.platforms = []
        self.devices = {}
        self.infos = {}
        self.contexts = {}
        self.restore()

    def __getstate__(self):
        """Do not pickle anything.
        """
        return {}

    def _create_contexts(self):
        """Creates OpenCL contexts.
        """
        for platform in self.platforms:
            self.contexts[platform] = pyopencl.Context(self.devices[platform])

    def restore(self, do_tests = 1):
        """Initialize an object after restoring from snapshot.
        """
        self.platforms = pyopencl.get_platforms()
        for platform in self.platforms:
            self.devices[platform] = platform.get_devices()
            for device in self.devices[platform]:
                self.infos[device] = DeviceInfo(device)
        self._create_contexts()
        if do_tests:
            self.do_tests()

    def do_tests(self):
        """Measure relative device performance.
        """
        #TODO(a.kazantsev): add simple benchmark here & update device ratings.
        pass
