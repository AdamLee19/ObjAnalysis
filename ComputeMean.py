import Mesh, MeshIO

import numpy as np
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
    objDir = './objs'
    faceV = []
    meshData = []
    for o in os.listdir(objDir):
        if o.endswith('.obj'):
            mesh = Mesh.Mesh()
            MeshIO.readMesh(mesh, os.path.join(objDir, o))
            meshData.append(mesh)
            faceV.append(mesh.vertices)
            print(o, len(mesh.vertices))



    meshData[0].vertices = computeMean(faceV)
    MeshIO.writeMesh(meshData[0], "mean.obj")