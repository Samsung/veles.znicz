from zope.interface import implementer

from veles.opencl_units import OpenCLUnit, IOpenCLUnit
import veles.znicz  # pylint: disable=W0611


@implementer(IOpenCLUnit)
class TrivialOpenCLUnit(OpenCLUnit):
    def cpu_run(self):
        pass

    def ocl_run(self):
        pass
