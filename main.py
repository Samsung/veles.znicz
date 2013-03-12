#!/usr/bin/python3
"""
Created on Mar 11, 2013

Entry point.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import logging


def main():
    logging.debug("Entered")
    f = filters.ContainerFilter()
    aa = f.add(filters.All2AllFilter())
    bb = f.add(filters.All2AllFilter())
    f.link(aa, bb)
    try:
        f.link(aa, bb)  # Exception
    except filters.ErrExists:
        print("Exception: link already exists")
    print(f.__doc__)
    logging.debug("Finished")


if __name__ == '__main__':
    main()
