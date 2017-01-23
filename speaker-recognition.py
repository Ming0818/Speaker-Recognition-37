import numpy as np
import h5py
import matplotlib

from RingBuffer.ringBuffer import *


@coroutine
def h5readTestMfcc(next):
    (yield)

    with h5py.File("BFMTV_CultureEtVous_2012-11-16_064700.h5", "r") as fh:
        data = fh.get("cep").value
        energy = fh.get("energy").value
        label = fh.get("vad").value

    print("YEHE")
    print type(data)
    flow = Flow(np.array(data), np.array(energy), np.array(label))
    print type(flow.data)
    next.send(flow)

@coroutine
def calculR(next, millis):
    try:
        time = 0 #timestamp of current packet treated
        while(True):
            input = yield #received Flow value

            size = len(input.data)
            print type(input.data)
            print type(input.data[:size/2])
            g1 = GaussianModel(input.data[:size/2])
            g2 = GaussianModel(input.data[size/2:])
            R = genR()
            next.send(R)
    except GeneratorExit:
        next.close()


@coroutine
def trace(millis):
    cpt = 0
    buf = [None]*(10)
    while(True):
        input = yield
        if(cpt == 9):
            buf[cpt] = input
            cpt = 0
            print buf
        cpt += 1


class GaussianModel :
    def __init__(self, data):
        self.data = data
        self.mean = None
        self.cov = None
        self.det = None
        self.populate()

    def populate(self):
        #mean of data
        print "POPULATE"
        print type(self.data)
        print "POPULATE"
        self.mean = self.data.mean(axis = 0)
        #covariance
        print "data"
        print(self.data)
        self.cov = np.cov(self.data)
        #determinant
        print(self.cov)
        self.det = np.linalg.det(self.cov)
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


output = trace(10)
r = calculR(output, 10)
buf = ring_buffer(r, 4, 2)
source = h5readTestMfcc(buf)
try :
    next(source)
except StopIteration :
    print("That's all folks")


"""
data = np.array([[4,5,6],[77,-8,-85], [10,4,12], [13,-140,15], [16,107,-188]])
x1 = GaussianModel(data)
data = np.array([[4,55,6],[7,8,65], [-10,11,-54], [13,14,15], [-146,17,18]])
x2 = GaussianModel(data)
genR(x1, x2)
"""
