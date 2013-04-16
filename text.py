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
        self.TrainIndex = formats.Labels()
        self.ValidIndex = formats.Labels()
        self.TestIndex  = formats.Labels()
        

    def load_original(self):
        """ Loads data from original Wine dataset.
            we will read config file with parameters dataset(width,height),
             size(or %) train set,size(or %) validation set,size(or %) test set,
             random or on series or random with seed...
             type normalization dataset(range)
             
             
        """ 
        print("One time relatively slow load from original Wine Dataset...")
        

        self.labels.batch=filters.aligned_zeros([178])
        self.labels.n_classes=3
        self.output.batch=filters.realign(numpy.loadtxt("wine/wine.csv", numpy.float32).reshape([178,13]))
        self.labels.batch=filters.realign(numpy.loadtxt("wine/wine_y_labels.csv", numpy.float32).reshape([178,1]))
        self.labels.batch=filters.aligned_zeros([178])
        print("Done")

    def normalize_use_range_train (self,Range):
        """
        """
        self.normalize_x_use_range_train(self,Range)
        self.normalize_y_use_range_train(self,Range)
        
    def normalize_use_all_dataset (self):
        """
        """        
        self.normalize_x_use_all_dataset(self)
        self.normalize_y_use_all_dataset(self)
        
    def normalize_x_use_all_dataset(self):
        """ normalization  input data.
        """    
        self.outmean =filters.aligned_zeros([13])
        
        self.outmean= numpy.mean(self.output.batch, axis=0)
        
        self.outstd =filters.aligned_zeros([13])
        self.outstd= numpy.std(self.output.batch, axis=0)
        
        self.output2 =filters.realign(filters.aligned_zeros([178*13]).reshape([178,13]))
        self.output2[:]=self.output.batch[:]
        
        for i in range(0,13):
            self.output2[:,i]=((self.output.batch[:,i]-self.outmean[i]))/self.outstd[i]
        
        self.outmin =filters.aligned_zeros([13])
        self.outmin= numpy.min(self.output2, axis=0)
        
        self.outmax =filters.aligned_zeros([13])
        self.outmax= numpy.max(self.output2, axis=0)
        
        for i in range(0,13):
            self.output2[:,i]=((((self.output2[:,i]-self.outmin[i]))/(self.outmax[i]-self.outmin[i]))-0.5)*2
        
    def normalize_x_use_range_train (self,Range):
        """
        """    
    def denormalize_x(self):
        """
        """
    def normalize_y(self):
        """
        """
    def denormalize_y(self):
        """
        """
    def normalize_y_use_range_train (self,Range):
        """
        """            
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
