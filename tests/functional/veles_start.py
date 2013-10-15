#!/usr/bin/python3.3 -O
"""
Created on Oct 15, 2013
 test Veles for wine [2].

@author: Seresov Denis <d.seresov@samsung.com>
"""
import sys
import os


import pickle
def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)

if not this_dir:
    this_dir = "."
add_path("%s/../.." % (this_dir))
add_path("%s/../../../src" % (this_dir))


import veles_lib

def main():
    vel = veles_lib.veles();
    vel.initialize();
    vel.run()
    pass

if __name__ == "__main__":
    main()
    sys.exit(0)
