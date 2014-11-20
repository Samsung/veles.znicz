#!/usr/bin/python3 -O
import numpy
import os
import scipy.misc

imagenet_path = "/data/veles/datasets/FakeImagenet/Caffe"

n_rows = 256
n_cols = 256
n_images = 1000
n_classes = 1000


def main():
    for i in range(0, n_classes):
        out_path = os.path.join(imagenet_path, "%s" % i)
        try:
            os.mkdir(out_path, mode=0o775)
        except:
            pass
        train_path = os.path.join(out_path, "train")
        valid_path = os.path.join(out_path, "validation")
        try:
            os.mkdir(train_path, mode=0o775)
        except:
            pass
        try:
            os.mkdir(valid_path, mode=0o775)
        except:
            pass
        for j in range(0, n_images):
            pixels = numpy.random.randint(
                0, 256,
                n_rows * n_cols).astype(numpy.ubyte).astype(numpy.ubyte)
            image = pixels.astype(numpy.float32).reshape(n_rows, n_cols)
            scipy.misc.imsave(os.path.join(train_path, "image_%s.png" % j),
                              image)
            pixels = numpy.random.randint(
                0, 256,
                n_rows * n_cols).astype(numpy.ubyte).astype(numpy.ubyte)
            image = pixels.astype(numpy.float32).reshape(n_rows, n_cols)
            scipy.misc.imsave(os.path.join(valid_path, "image_%s.png" % j),
                              image)

if __name__ == "__main__":
    main()
