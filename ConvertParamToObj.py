from pathlib import Path
from NumpyIO import *
import MeshIO as mio
import Mesh



eigenVecDir = Path('./eigenVector-face.npy')
eigen_vec = readFile(eigenVecDir)
print(eigen_vec.shape)

meanDir = Path('mean-face.npy')
mean_vec = readFile(meanDir)
print(mean_vec.shape)


objDir = Path('./aligned/F_015.obj')
mesh = Mesh.Mesh()
mio.readMesh(mesh, objDir)

for n in Path('./npys/').glob('*.npy'):
		
	paramDir = Path(n)
	face_vec = readFile(paramDir).reshape(-1, 1)
	print(n,face_vec.shape)


	obj = eigen_vec @ face_vec + mean_vec
	

	mesh.updateVert(obj.reshape(-1,3).tolist())
	mio.writeMesh(mesh,  f"{n.with_suffix('.obj')}")
