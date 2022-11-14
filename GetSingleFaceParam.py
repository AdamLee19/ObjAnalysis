from pathlib import Path
import MeshIO as mio
import Mesh
import numpy as np
from NumpyIO import readFile, writeFile






eigenVecDir = Path('./eigenVector-face.npy')
eigen_vec = readFile(eigenVecDir)
print(eigen_vec.shape)

meanDir = Path('mean-face.npy')
mean_vec = readFile(meanDir)
print(mean_vec.shape)

objDir = Path('./aligned')
for o in objDir.glob('*.obj'):
    mesh = Mesh.Mesh()
    mio.readMesh(mesh, o)
    point_vec = np.array(mesh.vertices).flatten().reshape(-1, 1)
    
    param = eigen_vec.T @ (point_vec - mean_vec)
    print(o, point_vec.shape)

    writeFile(o.with_suffix('.npy'), point_vec)


    

