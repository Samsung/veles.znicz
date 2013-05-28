'''
Created on Apr 19, 2013

@author: Seresov Denis <d.seresov@samsung.com>
'''
import units

class Repeater(units.Unit):
    """Propagates notification if any of the inputs are active.
    """
    def __init__(self, unpickling = 0):
        super(Repeater, self).__init__(unpickling=unpickling)
        if unpickling:
            return

    def gate(self, src):
        return 1
