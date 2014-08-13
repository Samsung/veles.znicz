#!/usr/bin/python3

import json
import sys

if __name__ == "__main__":
    all = {}
    for arg in sys.argv[1:]:
        with open(arg, "r") as fin:
            all.update(json.load(fin))

    with open("result_det_test_0.json", 'w') as fout:
        json.dump(all, fout, indent=4)
