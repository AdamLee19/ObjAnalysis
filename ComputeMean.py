import Mesh, MeshIO

import numpy as np
from pathlib import Path
import os

# Takes a listOfFace [[[k]n]m]
# Turns to n * k * m
# n: # of Points of each face
# k: dimension
# m: how many faces
def computeMean(listOfFace):
    npForm = np.stack(listOfFace, axis = 2)
    meanFace = np.mean(npForm, axis=-1)
    return meanFace.tolist()
    


if __name__ == "__main__": 
    objDir = Path('./objs')
    faceV = []
    meshData = []
    for o in objDir.glob('*.obj'):
        mesh = Mesh.Mesh()
        MeshIO.readMesh(mesh, o)
        meshData.append(mesh)
        faceV.append(mesh.vertices)
        print(o, len(mesh.vertices))



    meshData[0].vertices = computeMean(faceV)
    MeshIO.writeMesh(meshData[0], "mean.obj")