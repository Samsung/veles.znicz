"""
Created on Mar 20, 2013

File for MNIST dataset.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import filters
import data_batch
import struct
import error
import pickle


class MNISTLoader(filters.GeneralFilter):
    """Loads MNIST data
    """
    def load_original(self):
        """Loads data from original MNIST files
        """
        print("One time slow reading and preprocessing of original MNIST data...")
        
        # Reading labels:
        fin = open("MNIST/train-labels.idx1-ubyte", "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2049:
            raise error.ErrBadFormat("Wrong header in train-labels")

        n_labels, = struct.unpack(">i", fin.read(4))
        if n_labels != 60000:
            raise error.ErrBadFormat("Wrong number of labels in train-labels")

        labels = fin.read(n_labels)
        if len(labels) != n_labels:
            raise error.ErrBadFormat("EOF reached while reading labels from train-labels")

        fin.close()

        # Reading images:
        fin = open("MNIST/train-images.idx3-ubyte", "rb")

        header, = struct.unpack(">i", fin.read(4))
        if header != 2051:
            raise error.ErrBadFormat("Wrong header in train-images")

        n_images, = struct.unpack(">i", fin.read(4))
        if n_images != n_labels:
            raise error.ErrBadFormat("Wrong number of images in train-images")

        n_rows, n_cols = struct.unpack(">2i", fin.read(8))
        if n_rows != 28 or n_cols != 28:
            raise error.ErrBadFormat("Wrong images size in train-images")

        data = fin.read(n_images * n_rows * n_cols)  # 0 - white, 255 - black
        if len(data) != n_images * n_rows * n_cols:
            raise error.ErrBadFormat("EOF reached while reading images from train-images")

        fin.close()

        # Transforming images into float arrays 32*32 (place in the center and normalize):
        self.output = data_batch.DataBatch2D(n_images, 32, 32, labels)
        
        # This works really slow (mostly due to the data type conversion between python and numpy)
        offs = 0
        for i_image in range(0, n_images - 1):
            min_vle = 1.0e30
            max_vle = -1.0e30
            for i_row in range(1, n_rows):
                for i_col in range(1, n_cols):
                    vle = (1.0 / 255.0) * data[offs]
                    if vle < min_vle:
                        min_vle = vle
                    if vle > max_vle:
                        max_vle = vle
                    self.output.data[i_image, i_row, i_col] = vle
                    offs += 1
            if min_vle != 0.0 or max_vle != 1.0:
                for i_row in range(1, n_rows):
                    for i_col in range(1, n_cols):
                        vle = self.output.data[i_image, i_row, i_col]
                        self.output.data[i_image, i_row, i_col] = (vle - min_vle) / (max_vle - min_vle)
        print("Done")
        
        print("Saving data to cache for later fast load...")
        fout = open("MNIST/train.pickle", "wb")
        pickle.dump(self.output, fout)
        fout.close()
        print("Done")
        
        
    def input_changed(self, src):
        """GeneralFilter method.
        
        Here we will load MNIST data.
        """
        try:
            fin = open("MNIST/train.pickle", "rb")
            self.output = pickle.load(fin)
            fin.close()
        except IOError:
            self.load_original()
        self.output.mtime += 1
        if self.parent:
            self.parent.output_changed(self)
