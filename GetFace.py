import Mesh
import MeshIO as mio
import numpy as np
from pathlib import Path
from NumpyIO import writeFile, readFile



    


if __name__ == "__main__": 
    objDir = Path('./mean.obj')
    mesh = Mesh.Mesh()
    mio.readMesh(mesh, objDir)

    faces = mesh.get_faces_triangle()

    faces = np.array(faces, dtype=int)
    writeFile('Faces.npy', faces)


        