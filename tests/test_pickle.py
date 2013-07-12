"""
Created on May 21, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import unittest
import units
import os
import pickle


g_pt = 0


class PickleTest(units.SmartPickler):
    """Pickle test.
    """
    def __init__(self, unpickling=0, a="A", b="B", c="C"):
        global g_pt
        g_pt += 1
        super(PickleTest, self).__init__(unpickling)
        if unpickling:
            return
        self.a = a
        self.b = b
        self.c = c


class TestPickle(unittest.TestCase):
    def test(self):
        # Test for correct behavior of units.SmartPickler
        pt = PickleTest(a="AA", c="CC")
        self.assertEqual(g_pt, 1, "Pickle test failed.")
        pt.d = "D"
        pt.h_ = "HH"
        try:
            os.mkdir("cache")
        except OSError:
            pass
        fout = open("cache/test.pickle", "wb")
        pickle.dump(pt, fout)
        fout.close()
        del(pt)
        fin = open("cache/test.pickle", "rb")
        pt = pickle.load(fin)
        fin.close()
        self.assertListEqual([g_pt, pt.d, pt.c, pt.b, pt.a, pt.h_],
                             [2, "D", "CC", "B", "AA", None],
                             "Pickle test failed.")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test']
    unittest.main()
