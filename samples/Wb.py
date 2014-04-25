#!/usr/bin/python3.3
"""
Created on Aug 21, 2013

Outputs first layer weights to stdout separated by ", ".

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("IN", help="input file")
    args = parser.parse_args()

    fin = open(args.IN, "rb")
    (W, b) = pickle.load(fin)
    fin.close()

    weights = W[0]
    for row in weights:
        print(", ".join("%.6f" % (x) for x in row))
