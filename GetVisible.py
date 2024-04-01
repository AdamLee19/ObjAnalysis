# import Mesh, MeshIO
import numpy as np
from pathlib import Path
from NumpyIO import writeFile, readFile
import MeshIO as mio
import Mesh

    


if __name__ == "__main__":
    camera = 'right'
    face_folder = 'Face00033'
    visiblePtDir = Path(f'./{face_folder}/{camera}_visible.npy')
    modelMatDir = Path(f'./{face_folder}/ModelMat.npy')
    paramDir = Path(f'./{face_folder}/PointParam.npy')
    objDir = Path('./mean.obj')

    cameraDir   = Path(f'./common/Camera/{camera}/view.npy')
    eigenVecDir = Path('./common/Point/EigenVector.npy')
    meanDir = Path('./common/Point/Mean.npy')

    visiblePt = readFile(visiblePtDir)
    print(visiblePt.shape)
   
    modelMat = readFile(modelMatDir)
    viewMat = readFile(cameraDir)
    param = readFile(paramDir)
    mean = readFile(meanDir)
    eigenVec = readFile(eigenVecDir)

    vmMat = viewMat @ modelMat
    obj = (eigenVec @ param + mean).reshape(-1, 3)
    obj = mean.reshape(-1, 3)
    

    finalPoints = obj[visiblePt==1]

    # numPoints = obj.shape[0]
    # ones = np.ones((numPoints, 1))
    # obj = np.hstack((obj, ones)).T
    # view = (vmMat @ obj)[:3, :].T
    # finalPoints = view[visiblePt]

    # numPoints = finalPoints.shape[0]
    # ones = np.ones((numPoints, 1))
    # finalPoints = np.hstack((finalPoints, ones)).T
    # finalPoints = (np.linalg.inv(vmMat) @ finalPoints)[:3, :].T  
    # print(finalPoints.shape)
    #np.linalg.inv(vmMat) @
    mesh = Mesh.Mesh()
    mio.readMesh(mesh, objDir)
    mesh.updateVert(obj.tolist())
    mio.writeMesh(mesh,  f"{'./npys/'+face_folder+'.obj'}")

    pointsList = finalPoints.tolist()
    v = list(map(lambda x: 'v {} {} {}\n'.format(x[0], x[1], x[2]),pointsList))

    with open(f'./npys/{camera}.obj', "w") as fp:
        fp.writelines(v)
        