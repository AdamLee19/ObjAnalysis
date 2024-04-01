import numpy as np
from NumpyIO import *

total_points = 12466
with open('mouthSocket.txt') as f:
    

    idx = f.read().strip().split('\n')
    idx = list(map(int, idx))
    
    idx = np.array(idx, dtype=int)

    finalIdx = np.zeros(total_points, dtype=int)
    finalIdx[idx] = 1

    writeFile('Sockets.npy', finalIdx)