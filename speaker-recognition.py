import numpy as np
import h5py
#import matplotlib
import matplotlib.pyplot as plt

from RingBuffer.ringBuffer import *


@coroutine
def h5readTestMfcc(next):
    try:
        (yield)
        with h5py.File("BFMTV_CultureEtVous_2012-11-16_064700.h5", "r") as fh:
            data = fh.get("cep").value
            energy = fh.get("energy").value
            label = fh.get("vad").value

        flow = Flow(np.array(data), np.array(energy), np.array(label))
        next.send(flow)
    except GeneratorExit:
        next.close()

@coroutine
def calculR(next, millis):
    try:
        time = 0 #timestamp of current packet treated
        while(True):
            input = yield #received Flow value

            size = len(input.data)
            g1 = GaussianModel(input.data[:size//2])
            g2 = GaussianModel(input.data[size//2:])
            g = GaussianModel(input.data)
            R = genR(g1, g2, g)
            #print(R)
            next.send(R)
    except GeneratorExit:
        next.close()


#Take initial time and the windows of time
@coroutine
def trace(starttime_ms, gap_ms, window_ms):
    try:
        value = window_ms // gap_ms
        time = starttime_ms
        cpt = 0
        x = [None]*value
        y = [None]*value
        #buf = {}
        #while(True):

        while(cpt < value):
            input = yield
            #reducing the highest values
            if (input > pow(10, 10)):
                input = pow(10, 10)
            x[cpt] = time
            y[cpt] = input
            cpt += 1
            time += gap_ms
        #after loop
        if (cpt == value):
            plt.plot(x, y)
            plt.show()
            cpt +=1
    except GeneratorExit:
        pass

@coroutine
#Take initial time and the windows of time
def detect(starttime_ms, gap_ms, window_ms):
    try:
        value = window_ms // gap_ms
        time = starttime_ms
        limit = pow(10, 10)
        cpt = 0
        #buf = {}
        #while(True):

        while(True):
            input = yield
            #reducing the highest values

            sec = "%02d" % ((time//1000)%60) #current time in seconds

            if (input > limit):
                input = limit
                print("R: {}m{} : {}".format( (time//60000), sec, input) )
            cpt += 1
            time += gap_ms

            if(window_ms + starttime_ms <= time):
                break
    except GeneratorExit:
        pass



class GaussianModel :
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data
        self.mean = None
        self.cov = None
        self.logdet = None
        self.populate()

    def populate(self):
        #mean of data
        self.mean = self.data.mean(axis = 0)
        #print "DATA:"
        #print(self.data)
        #covariance
        self.cov = np.cov(self.data)
        #print("COV:")
        #print(self.cov)
        #determinant
        (sign, self.logdet) = np.linalg.slogdet(self.cov)
        #print("DET:", self.det)

#equ. 2.30
def genR(x1, x2, x0):
    """
    data = np.concatenate((x1.data, x2.data), axis=0)
    print("concat ", data)
    x0 = GaussianModel([x1.data, x2.data])
    print("Gaussian ", x0)
    """
    GLR = ((len(x0.data)/2) * x0.logdet) - (((len(x1.data)/2) * x1.logdet) + ((len(x2.data)/2) * x2.logdet))
    #print("GLR: ", GLR)
    #R = np.exp(-np.log(10) * GLR)
    #print("R ", R)
    return GLR

def deltaBIC(x1, x2):
    data = np.concatenate((x1.data, x2.data), axis=0)
    #print("concat ", data)


nb_ech = 250
gap_ms = 10
window_calcul = 60000 #en ms

output = trace(nb_ech*gap_ms+gap_ms, gap_ms, window_calcul) #trace or detect
r = calculR(output, gap_ms)
buf = ring_buffer(r, nb_ech*2, nb_ech*2-1)
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
