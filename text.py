'''
Created on Apr 16, 2013

@author: Seresov Denis <d.seresov@samsung.com>
'''

import filters
import formats
import pickle
import numpy


class TXTLoader(filters.Filter):
    """Loads Wine data.

    State:
        output: contains Wine training set data
        labels: contains wine training set labels.
    """
    def __init__(self, unpickling = 0):
        super(TXTLoader, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.output = formats.Batch()
        self.labels = formats.Labels()

    def load_original(self):
        """ Loads data from original Wine dataset.
        """
        print("One time relatively slow load from original Wine Dataset...")

        self.labels.batch=filters.aligned_zeros([178])
        self.labels.n_classes=3
        self.output.batch=filters.realign(numpy.loadtxt("wine/wine.csv", numpy.float32).reshape([178,13]))
        self.labels.batch=filters.realign(numpy.loadtxt("wine/wine_y_labels.csv", numpy.float32).reshape([178,1]))
           
        
        
        print("Done")
    def initialize(self):
        """Here we will load Wine data.
        """
        
        try:
            fin = open("cache/Wine-train.pickle", "rb")
            self.output.batch, self.labels.batch, self.labels.n_classes = pickle.load(fin)
            fin.close()
        except IOError:
            self.load_original()
        self.output.update()

    def run(self):
        """Just update an output.
        """
        self.output.update()
