'''
Created on Apr 16, 2013

@author: Seresov Denis <d.seresov@samsung.com>
'''

import units
import formats
import numpy
import config


class TXTLoader(units.Unit):
    """Loads Wine data.

    State:
        output: contains Wine training set data
        labels: contains wine training set labels.
        a: target mse.
    """
    def __init__(self, unpickling=0):
        super(TXTLoader, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.output = formats.Batch()
        self.output2 = formats.Batch()
        self.labels = formats.Labels()
        self.TrainIndex = formats.Labels()
        self.ValidIndex = formats.Labels()
        self.TestIndex = formats.Labels()

    def normalize_use_all_dataset(self):
        print("normalize_use_all_dataset start")
        self.normalize_x_use_all_dataset()
        #self.normalize_y_use_all_dataset()
        print("normalize_use_all_dataset ok")

    def load_original(self):
        """ Loads data from original Wine dataset.
            we will read config file with parameters dataset(width,height),
            size(or %) train set,size(or %) validation set,size(or %) test set,
            random or on series or random with seed...
            type normalization dataset(range)
        """
        print("Load from original Wine Dataset...")

        self.labels.batch = numpy.zeros([178],
                                        dtype=config.dtypes[config.dtype])
        self.labels.n_classes = 3
        self.output.batch = numpy.loadtxt("wine/wine.csv",
            config.dtypes[config.dtype]).reshape([178, 13])
        self.labels.batch = numpy.loadtxt("wine/wine_y_labels.csv", numpy.int)
        self.normalize_use_all_dataset()
        print("Done")

    def normalize_use_range_train(self, Range):
        self.normalize_x_use_range_train(self, Range)
        self.normalize_y_use_range_train(self, Range)

    def normalize_x_use_all_dataset(self):
        """ normalization  input data.
        """
        #self.outmean = numpy.zeros([13], dtype=config.dtypes[config.dtype])
        print("normalize_x_use_all_dataset start")

        self.outmean = numpy.mean(self.output.batch, axis=0)

        self.outstd = numpy.std(self.output.batch, axis=0)

        self.output2.batch = numpy.zeros([178 * 13],
            dtype=config.dtypes[config.dtype]).reshape([178, 13])
        self.output2.batch[:] = self.output.batch[:]

        for i in range(0, 13):
            self.output2.batch[:, i] = ((self.output.batch[:, i] -
                                         self.outmean[i])) / self.outstd[i]

        #self.outmin = numpy.zeros([13], dtype=config.dtypes[config.dtype])
        self.outmin = numpy.min(self.output2.batch, axis=0)

        #self.outmax = numpy.zeros([13], dtype=config.dtypes[config.dtype])
        self.outmax = numpy.max(self.output2.batch, axis=0)

        for i in range(0, 13):
            self.output2.batch[:, i] = ((((self.output2.batch[:, i] -
                                           self.outmin[i])) /
                                        (self.outmax[i] - self.outmin[i])) -
                                        0.5) * 2

        print(self.output2.batch.size)
        print("normalize_x_use_all_dataset ok")

    def normalize_x_use_range_train(self, Range):
        pass

    def denormalize_x(self):
        pass

    def normalize_y(self):
        pass

    def denormalize_y(self):
        pass

    def normalize_y_use_range_train(self, Range):
        pass

    def initialize(self):
        """Here we will load Wine data.
        """
        self.load_original()
        self.output2.update()
        print(self.output2.batch.size)

    def run(self):
        """Just update an output.
        """
        self.output2.update()
