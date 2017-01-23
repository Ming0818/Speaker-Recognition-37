
import numpy as np
from scipy import linalg

from RingBuffer.ringBuffer import *

class GaussianModel :
    def __init__(self, data):
        self.data = data
        self.mean = None
        self.cov = None
        self.det = None
        self.populate()

    def populate(self):
        #mean of data
        self.mean = data.mean(axis = 0)
        #covariance
        self.cov = np.cov(data)
        #determinant
        self.det = linalg.det(self.cov)
        print("det ", self.det)

#equ. 2.30
def genR(x1, x2):
    data = np.concatenate((x1.data, x2.data), axis=0)
    print("concat ", data)
    x0 = GaussianModel([x1.data, x2.data])
    print("Gaussian ", x0)
    GLR = ((len(x0.data)/2) * np.log10(x0.det)) - (((len(x1.data)/2) * np.log10(x1.det)) + ((len(x2.data)/2) * np.log10(x2.det)))
    print("GLR ", GLR)
    R = np.exp(-np.log(10) * GLR)
    print("R ", R)

def deltaBIC(x1, x2):
    data = np.concatenate((x1.data, x2.data), axis=0)
    print("concat ", data)


data = np.array([[4,5,6],[77,-8,-85], [10,4,12], [13,-140,15], [16,107,-188]])
x1 = GaussianModel(data)
data = np.array([[4,55,6],[7,8,65], [-10,11,-54], [13,14,15], [-146,17,18]])
x2 = GaussianModel(data)
genR(x1, x2)
