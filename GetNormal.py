# import Mesh, MeshIO
import numpy as np
from pathlib import Path
from src.utils.NumpyIO import writeFile, readFile
from src.utils.pca import PCA
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
    normalDir   = Path('./face/common/VertexNormal/VertexNormal.npy')
    cameraDir   = Path('./face/common/Camera/left/view.npy')
    modelMatDir = Path('./Face03123/ModelMat.npy')
    paramDir = Path('./Face03123/PointParam.npy')
    eigenVecDir = Path('./face/common/Point/EigenVector.npy')
    meanDir = Path('./face/common/Point/Mean.npy')

    normal = readFile(normalDir)
    modelMat = readFile(modelMatDir)
    viewMat = readFile(cameraDir)
    param = readFile(paramDir)
    mean = readFile(meanDir)
    eigenVec = readFile(eigenVecDir)
    
    obj = (eigenVec @ param + mean).reshape(-1, 3)
    numPoints = obj.shape[0]
    
    ones = np.ones((numPoints, 1))
    obj = np.hstack((obj, ones)).T
    zeros = np.zeros((numPoints, 1))
    normal = np.hstack((normal, zeros)).T
   


    vmMat = viewMat @ modelMat

    view = -(vmMat @ obj)[:3, :].T
    view = view / np.linalg.norm(view, axis=1)[:, np.newaxis]
    normal = (vmMat @ normal)[:3, :].T
    normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
    result = np.sum(view * normal, axis=1)
    idices = np.where(result>0)
    finalPoints =  np.squeeze(obj[:3, :].T[idices, :], axis=0)
    
    pointsList = finalPoints.tolist()
    v = list(map(lambda x: 'v {} {} {}\n'.format(x[0], x[1], x[2]),pointsList))

    with open('test.obj', "w") as fp:
        fp.writelines(v)
        