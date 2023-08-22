from pathlib import Path
from NumpyIO import *
import MeshIO as mio
import Mesh
import numpy



numpy.set_printoptions(suppress=True) 
		
paramDir = Path('./npys/PointParam.npy')
face_vec = readFile(paramDir).reshape(48,)
print(face_vec)



predicDir = Path('./npys/Face1738-100.npy')
pred_vec = readFile(predicDir).astype(float)
print(pred_vec)


# for i in range(15):
# 	pred_vec[i] = face_vec[i]
# writeFile(predicDir, pred_vec)

# sinV = readFile(Path('singularValue-face.npy'))
# sinV = sinV ** 2 / 9

# print(sinV)
# print(sinV / numpy.sum(sinV, axis = 0))
