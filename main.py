#!/usr/bin/python3
"""
Created on Mar 11, 2013

Entry point.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters


def main():
    f = filters.All2AllFilter()
    print(f.__doc__)
    print("End of job")


if __name__ == '__main__':
    main()
