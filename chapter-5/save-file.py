import numpy as np

from htm.bindings.sdr import SDR
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters
from htm.algorithms import SpatialPooler as SP
from htm.algorithms import TemporalMemory as TM
from htm.bindings.algorithms import Classifier

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import pickle

pitchNames = ("C","D","E","F","G","A","B")

scalarEncoderParams = ScalarEncoderParameters()
scalarEncoderParams.minimum = 0
scalarEncoderParams.maximum = 6
scalarEncoderParams.activeBits = 3
scalarEncoderParams.category = True

enc = ScalarEncoder(scalarEncoderParams)

print ("C: ", enc.encode(0))
print ("D: ", enc.encode(1))
print ("E: ", enc.encode(2))
print ("F: ", enc.encode(3))
print ("G: ", enc.encode(4))
print ("A: ", enc.encode(5))
print ("B: ", enc.encode(6))

inputSDR  = SDR( dimensions = (21, ) )
activeSDR = SDR( dimensions = (576,) )
sp = SP(inputDimensions  = inputSDR.dimensions,
        columnDimensions = activeSDR.dimensions,
        localAreaDensity = 0.02,
        globalInhibition = True,
        seed             = 1,
        synPermActiveInc   = 0.01,
        synPermInactiveDec = 0.008)

print(sp)

clsr = Classifier()

tm = TM(
    columnDimensions = (576,),
    cellsPerColumn=8,
    initialPermanence=0.5,
    connectedPermanence=0.5,
    minThreshold=8,
    maxNewSynapseCount=20,
    permanenceIncrement=0.1,
    permanenceDecrement=0.0,
    activationThreshold=8,
)
print(tm)

for i in range(len(pitchNames)):
    inputSDR = enc.encode(i)
    print("input SDR: ",inputSDR)
    sp.compute(inputSDR, True, activeSDR)
    print("Active SDR: ",activeSDR)
    clsr.learn( activeSDR.addNoise(0.2), i )
    print("Classifier learn: ",i)
    print("")

seq=[0,0,4,4,5,5,4,3,3,2,2,1,1,0]

batch=10
for n in range(batch):    
    for i in range(len(seq)):
        inputSDR = enc.encode(seq[i])
        print("input SDR: ",inputSDR)
        sp.compute(inputSDR, True, activeSDR)
        print("Active SDR: ",activeSDR)
        print("")

        tm.compute( activeSDR, learn=True)
        tm.activateDendrites(True)

        activeColumnsIndices   = [tm.columnForCell(i) for i in tm.getActiveCells().sparse]
        predictedColumnIndices = [tm.columnForCell(i) for i in tm.getPredictiveCells().sparse]

        print(tm.getActiveCells())
        print(tm.getPredictiveCells())

        active_sdr = SDR( tm.numberOfColumns() )
        active_sdr.sparse  = np.array(sorted(set(activeColumnsIndices)))
        print("-"*70 )
        print("Active sdr: ", active_sdr)

        predict_sdr = SDR( tm.numberOfColumns() )
        predict_sdr.sparse  = np.array(sorted(set(predictedColumnIndices)))
        print("-"*70 )
        print("Predicted sdr: ",predict_sdr)

        if  len(predict_sdr.sparse) == 0:
            tm_predict = "nan"
            tm_pitchNames = "nan"      
        else:
            tm_predict = np.argmax( clsr.infer( predict_sdr) ) 
            tm_pitchNames = pitchNames[tm_predict]

        print("-"*70 )
        print("PDF: ",clsr.infer( predict_sdr) )
        print("-"*70 )

        print("predict index: ",tm_predict)
        print("-"*70 )
        print("predict label: ",tm_pitchNames)

        print("")
      
tm.saveToFile("tm.model")