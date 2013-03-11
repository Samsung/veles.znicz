#!/usr/bin/python3
"""
Created on Mar 11, 2013

Entry point.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import layer


def main():
    l = layer.Layer()
    print(l.__doc__)
    print("End of job")


if __name__ == '__main__':
    main()
