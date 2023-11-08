from pathlib import Path
from NumpyIO import *
import MeshIO as mio
import Mesh
import numpy as np




np.set_printoptions(precision=8, suppress=True)

eigenVecDir = Path('./common/Point/EigenVector.npy')
eigen_vec = readFile(eigenVecDir)[:,:-1]
print(eigen_vec.shape)

meanDir = Path('./common/Point/Mean.npy')
mean_vec = readFile(meanDir)
print(mean_vec.shape)


eigenValDir = Path('./common/Point/EigenValue.npy')
eigen_val = readFile(eigenValDir).flatten()
print(np.sqrt(eigen_val))


print('\n\n\n\n')
objDir = Path('./mean.obj')
mesh = Mesh.Mesh()
mio.readMesh(mesh, objDir)

fp = open('./common/ibug68.txt', 'r')
a = fp.readlines()
fp.close()
lm_idx = [int(l) for l in a]

for n in Path('./npys/').glob('*.npy'):
		
	paramDir = Path(n)
	face_vec = readFile(paramDir).reshape(-1, 1)
	dim, _ = face_vec.shape
	if dim == 48: face_vec = face_vec[:-1,:]
	
	print(n,face_vec.shape)
	print(face_vec.flatten())

	
	obj = eigen_vec @ face_vec + mean_vec

	print(obj.shape)
	

	mesh.updateVert(obj.reshape(-1,3).tolist())
	mio.writeMesh(mesh,  f"{n.with_suffix('.obj')}")

