import numpy as np
from numpy.core.fromnumeric import mean
from pathlib import Path
import MeshIO as mio
import Mesh






alignedDir = Path('./aligned') 
faceV = []
for o in alignedDir.glob('*.obj'):
    print(o.name)
    mesh = Mesh.Mesh(name=o.name)
    mio.readMesh(mesh, o)
    faceV.append(np.array(mesh.vertices).flatten())
    



training = np.array(faceV).T

mean_face = np.mean(training, axis = 1).reshape(-1, 1)

trainSubMean = training - mean_face

print("Compute SVD")
u, s, vh = np.linalg.svd(trainSubMean, full_matrices=False)

param = u.T @ trainSubMean



print("Saving Eigen Vector U {}".format(u.shape))
with open('eigenVector-face.npy', 'wb') as f:
    np.save(f, u) # (37398, 48)


print("Saving face matrix {}".format(training.shape))
with open('face.npy', 'wb') as f:
    np.save(f, training)  #(37398, 48)


print("Saving paramter matrix param {}".format(param.shape))
with open('param-face.npy', 'wb') as f:
    np.save(f, param)  #(48, 48)

s = s.reshape(-1,1)
print("Saving eigenValue matrix s {}".format(s.shape))
with open('eigenValue-face.npy', 'wb') as f:
    np.save(f, s)  # (48, 1)


print("Saving mean {}".format(mean_face.shape))
with open('mean-face.npy', 'wb') as f:
    np.save(f, mean_face)  # (37398 , 1)