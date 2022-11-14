from pathlib import Path
from NumpyIO import *
import MeshIO as mio
import Mesh

paramDir = Path('param.npy')
face_vec = readFile(paramDir)
print(face_vec.shape)


eigenVecDir = Path('./eigenVector-face.npy')
eigen_vec = readFile(eigenVecDir)
print(eigen_vec.shape)

meanDir = Path('mean-face.npy')
mean_vec = readFile(meanDir)
print(mean_vec.shape)


objDir = Path('./aligned/F_015.obj')
mesh = Mesh.Mesh()
mio.readMesh(mesh, objDir)


obj = eigen_vec @ face_vec + mean_vec
print(obj.shape)

mesh.updateVert(obj.reshape(-1,3).tolist())
mio.writeMesh(mesh,  'new.obj')
