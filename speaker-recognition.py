# -*- coding: utf-8 -*-

import numpy as np
import h5py
#import matplotlib
import matplotlib.pyplot as plt

from RingBuffer.ringBuffer import coroutine
from RingBuffer.ringBuffer import Flow
from RingBuffer.ringBuffer import ring_buffer as ringBufferFlow
from RingBufferGeneric.ringBuffer import ring_buffer as ringBufferGeneric


@coroutine
def h5readTestMfcc(next):
    try:
        (yield)
        with h5py.File("BFMTV_CultureEtVous_2012-11-16_064700.h5", "r") as fh:
            data = fh.get("cep").value
            energy = fh.get("energy").value
            label = fh.get("vad").value

        #keep 12 first mfcc out of 19
        a = np.array(data)
        b = np.zeros((len(a), 12))
        for i in range(len(a)):
            b[i] = a[i][:12]

        flow = Flow(np.array(b), np.array(energy), np.array(label))
        next.send(flow)
    except GeneratorExit:
        next.close()

@coroutine
def calculR(next, millis, trace_ouput=None):
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
            if(trace_ouput is not None):
                trace_ouput.send(R)
    except GeneratorExit:
        next.close()

@coroutine
def minimumLocaux(starttime_ms, gap_ms):
    firstVal = True
    minIndex = 0
    minVal = 0
    index = 0
    time = starttime_ms
    minlocs = dict()

    try:
        input = yield
        #recherche du premier minimum local
        for i in range(len(input)):
            if(input[i] < minVal or firstVal):
                firstVal = False
                minVal = input[i]
                minIndex = i + index
        minlocs[minIndex] = (starttime_ms + (minIndex*gap_ms), minVal) #on stocke dans le dictionnaire un couple temps / valeur
        print("First minloc : {}".format(minlocs[minIndex]))

        while(True):
            index = index + 1
            input = yield

            #Si la plus petite valeur sort du champs, on en cherche une nouvelle
            if(minIndex < index):
                firstVal = True #la première valeur n'aura pas à être comparé
                print("Last value has rolled out")
                for i in range(len(input)):
                    if(input[i] < minVal or firstVal):
                        minVal = input[i]
                        minIndex = i + index
                        firstVal = False
            else: #sinon on regarde si la nouvelle valeur est plus petite que celle enregistrée
                if input[-1] < minVal:
                    minVal = input[-1]
                    minIndex = len(input)-1 + index

            if not (minIndex in minlocs.keys()):
                minlocs[minIndex] = (starttime_ms + (minIndex*gap_ms), minVal) #on stocke dans le dictionnaire un couple temps / valeur
                print("New minloc : {}".format(minlocs[minIndex]))
    except GeneratorExit:
        print minlocs
        pass

#Take initial time, time offset between values and the size fo window to plot
@coroutine
def trace(starttime_ms, gap_ms, window_ms):
    plot = plt
    value = window_ms // gap_ms
    time = starttime_ms
    cpt = 0
    x = [0]*value
    y = [0]*value

    try:
        while(True):
            input = yield
            x[cpt] = time
            y[cpt] = input
            cpt += 1
            time += gap_ms
            #print plot
            if (cpt == value):
                plt.plot(x, y)
                plt.show()
                cpt = 0
                x = [0]*value
                y = [0]*value

            #print process indicator every second
            if(time % 1000 < gap_ms):
                print("Processing: {}ms".format(time))

    except GeneratorExit:
        #print(x, y)
        plot.plot(x, y)
        plot.show()
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


nb_ech = 250 #nombre de mfcc de chaque côté #default 250
step_ech = 1 #décalage dans le buffer circulaire entre chaque série de mfcc #default 1
gap_ms = 10 #temps entre chaque mffc
window_calcul = 10000 #temps à traiter en ms #593000

time_minloc = 10000; #taille de la fenêtre pour les minimums locaux en ms

trace_output = trace(nb_ech*gap_ms+gap_ms, gap_ms*step_ech, window_calcul) #trace or detect
minloc = minimumLocaux(nb_ech*gap_ms, gap_ms*step_ech)
rbuf = ringBufferGeneric(minloc, time_minloc//(gap_ms//step_ech), time_minloc//(gap_ms//step_ech) - 1)
r = calculR(rbuf, gap_ms, trace_output)
buf = ringBufferFlow(r, nb_ech*2, nb_ech*2 - step_ech)
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
