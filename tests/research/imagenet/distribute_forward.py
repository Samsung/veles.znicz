#!/usr/bin/python3

import datetime
import pickle
import sys

if __name__ == "__main__":
    stats = []
    total = 0
    img_count = 0
    print("Loading %s..." % sys.argv[1])
    with open(sys.argv[1], "rb") as fin:
        while True:
            try:
                img = pickle.load(fin)[1]
                size = len(img["bbxs"])
                stats.append(size)
                total += size
                img_count += 1
            except EOFError:
                break
    print("Found %d images, %d bboxes" % (img_count, total))
    slaves = int(sys.argv[2])
    norm = total // slaves
    print("%d  bboxes for each slave" % norm)
    print("### ESTIMATED TIME: %s ###" % datetime.timedelta(
        seconds=(norm / 50000 * 1.5 * 3600)))
    print("")
    current_sum = 0
    border = 0
    minmaxs = []
    for index, count in enumerate(stats):
        current_sum += count
        if current_sum >= norm:
            print("[%d, %d)" % (border, index + 1))
            minmaxs.append((border, index + 1))
            border = index + 1
            current_sum = 0
    if border < len(stats):
        print("[%d, 0)" % border)
        minmaxs.append((border, 0))

    print('')
    print('-' * 80)
    print('')
    for index, minmax in enumerate(minmaxs):
        print(index)
        print('scripts/velescli.py -p "" -s -d 0:1 --debug MergeBboxes  '
              'veles/znicz/tests/research/imagenet/imagenet_forward.py - '
              'root.loader.min_index=%d root.loader.max_index=%d' % minmax)
