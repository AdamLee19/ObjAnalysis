from scipy.linalg import orthogonal_procrustes
from ComputeMean import computeMean
import MeshIO as mio
import Mesh
import numpy as np
from pathlib import Path



objDir = Path('./objs')
saveDir = Path('./objScaled')
saveDir.mkdir(exist_ok=True)

meshData = []
faceV = []
numPoints = 0
for o in objDir.glob('*.obj'):
    mesh = Mesh.Mesh(name=o.name)
    mio.readMesh(mesh, o)
    mesh.scaleMesh(ratio=100)
    meshData.append(mesh)
    print(o.name, len(mesh.vertices))




for m in meshData: mio.writeMesh(m, saveDir / m.name)



