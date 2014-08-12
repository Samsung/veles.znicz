#!/usr/bin/python3

import json
import pickle
import sys

if __name__ == "__main__":
    meta = None
    with open(sys.argv[1], 'rb') as fin:
        img = ""
        while img.find(sys.argv[2]) < 0:
            img, meta = pickle.load(fin)
    with open("extracted_%s.json" % sys.argv[2], 'w') as fout:
        json.dump({img: meta}, fout, indent=4)
