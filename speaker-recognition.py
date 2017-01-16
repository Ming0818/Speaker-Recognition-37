
import numpy as np

from RingBuffer.ringBuffer import *

class GaussianModel :
    def __init__(self, data):
        self.data = data
        self.mean = None
        self.covariance = None
        self.populate()

    def populate(self):
        #mean of data
        mean = data.mean(axis = 0)
        #print(data.mean(axis = 0))

        #covariance
        print(np.cov(data))



data = np.array([[4,5,6],[7,8,9], [10,11,12], [13,14,15], [16,17,18]])
tmp = GaussianModel(data)
