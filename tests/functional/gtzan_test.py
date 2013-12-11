#!/usr/bin/python3.3 -O
import numpy
import pickle
import sys
import re
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, required=True,
                        help="Snapshot with trained network weights and bias.")
    parser.add_argument("-window_size", type=int, required=True,
                        help="Size of the scanning window.")
    args = parser.parse_args()

    fin = open(args.snapshot, "rb")
    (W, b) = pickle.load(fin)
    fin.close()

    print("Snapshot loaded, will load dataset...")

    fin = open("/data/veles/music/GTZAN/gtzan.pickle", "rb")
    data = pickle.load(fin)
    fin.close()

    print("Dataset loaded, will check nn performance now...")

    labels = {"blues": 0,
              "country": 1,
              "jazz": 2,
              "pop": 3,
              "rock": 4,
              "classical": 5,
              "disco": 6,
              "hiphop": 7,
              "metal": 8,
              "reggae": 9}
    i_labels = {}
    for k, v in labels.items():
        i_labels[v] = k

    window_size = args.window_size
    features = ["Energy", "Centroid", "Flux", "Rolloff", "ZeroCrossings"]

    norm_add = {'Rolloff': (-4201.6782581196312),
                'Centroid': (-2030.0246076732176),
                'ZeroCrossings': (-55.139632154568375),
                'Flux': (-0.91916753884777358),
                'Energy': (-10437031.030474659)}
    norm_mul = {'Rolloff': 0.00016525803387664815,
                'Centroid': 0.00014463597973502374,
                'ZeroCrossings': 0.002526143259661856,
                'Flux': 0.066172350633377536,
                'Energy': 3.2782483475256631e-09}

    inp = numpy.zeros(len(features) * window_size, dtype=numpy.float64)

    n_ok = 0
    n_fail = 0

    lbl_re = re.compile("/(\w+)\.\w+\.\w+$")
    outs = numpy.zeros(len(labels), dtype=numpy.float64)
    for fnme, fvle in data["files"].items():
        limit = 2000000000
        ff = fvle["features"]
        res = lbl_re.search(fnme)
        lbl = labels[res.group(1)]
        outs[:] = 0
        for k in features:
            v = ff[k]["value"]
            limit = min(len(v), limit)
            v += norm_add[k]
            v *= norm_mul[k]
        for offs in range(0, limit - window_size + 1, window_size):
            offs2 = offs + window_size
            j = 0
            for k in features:
                v = ff[k]["value"]
                jj = j + window_size
                inp[j:jj] = v[offs:offs2]
                j = jj
            if inp.min() < -1 or inp.max() > 1:
                raise Exception("input is out of range")
            a = inp
            for i in range(len(W) - 1):
                weights = W[i]
                bias = b[i]
                out = numpy.dot(a, weights.transpose())
                out += bias
                out *= 0.6666
                numpy.tanh(out, out)
                out *= 1.7159
                a = out
            i = len(W) - 1
            weights = W[i]
            bias = b[i]
            out = numpy.dot(a, weights.transpose())
            out += bias
            # Apply softmax
            m = out.max()
            out -= m
            numpy.exp(out, out)
            smm = out.sum()
            out /= smm
            # Sum totals
            outs += out
        mx = numpy.argmax(outs)
        if mx == lbl:
            n_ok += 1
        else:
            n_fail += 1
            print("FAIL: %s as %s" % (fnme, i_labels[mx]))

    print("")
    print("n_ok: %d" % (n_ok))
    print("n_fail: %d" % (n_fail))
    print("%.2f%% errors" % (100.0 * n_fail / (n_ok + n_fail)))


if __name__ == '__main__':
    main()
    sys.exit(0)
