from scipy.linalg import orthogonal_procrustes
from ComputeMean import computeMean
import MeshIO as mio
import Mesh
import numpy as np
from pathlib import Path



objDir = Path('./objs')
saveDir = Path('./aligned')

meshData = []
faceV = []
numPoints = 0
for o in objDir.glob('*.obj'):
    mesh = Mesh.Mesh(name=o.name)
    mio.readMesh(mesh, o)
    meshData.append(mesh)
    faceV.append(mesh.vertices)
    print(o.name, len(mesh.vertices))




# construct an empty mean face for compare
numPoints = len(meshData[0].vertices)
meanPre = np.zeros((numPoints,3))
while True:
    meanNow = np.array(computeMean(faceV))
    rmse = np.mean(np.linalg.norm(meanPre - meanNow, axis=1))
    print("Root Mean Squared Error:", rmse)
    if np.isclose(rmse, 0, atol=1e-16): break
    for idx, m in enumerate(meshData):
        R, _ = orthogonal_procrustes(np.array(m.vertices), np.array(meanNow))
        m.vertices = (np.array(m.vertices) @ R).tolist()
        faceV[idx] = m.vertices
    meanPre = meanNow

for m in meshData: mio.writeMesh(m, saveDir / m.name)

meshData[0].vertices = computeMean(faceV)
mio.writeMesh(meshData[0], "mean.obj")


