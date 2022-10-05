import numpy as np
from functools import reduce

def readMesh(mesh, fileName):
    with open(fileName, "r") as fp:
        lines = fp.read().splitlines()
        for l in lines:
            if not l: continue #ignore empty lines

            if l[0] == '#': continue #ignore comments     



            l = l.split()
            elem = l.pop(0)

            if elem == 'v':
                # vertice format:[[x,y,z], [x,y,z]. [x,y,z]....]
                mesh.vertices.append(list(map(float, l)))
            elif elem == 'f':
                mesh.faces.append(l)
            elif elem == 'vn':
                mesh.normal.append(list(map(float, l)))
            elif elem == 'vt':
                mesh.texCoord.append(list(map(float, l)))
        
   
        


   
    
     

"""
    normalizedCoord: clip coordinate between -1 to 1
    coord: coordinate of a point (x, y, z)
    bbox: bounding box of the object. it's a list [xMin, xMax, yMin, yMax, zMin, zMax]
"""
def normalizeCoord(coord, bbox):
    point = np.array(coord)
    bboxMin = np.array([bbox[0], bbox[2], bbox[4]]).reshape(1, 3)
    bboxMax = np.array([bbox[1], bbox[3], bbox[5]]).reshape(1, 3)
   
   

    point = 2 * (point - bboxMin) / (bboxMax - bboxMin) - 1

    return (point.tolist())

def writeMesh(mesh, fileName):
    '''
        These two functions are a little bit fancy
        I am trying to learn these two functions
    '''
    v = list(map(lambda x: 'v {} {} {}\n'.format(x[0], x[1], x[2]), mesh.vertices))
    f = list(map(lambda x: 'f ' + reduce(lambda f1, f2: f1 + ' ' + f2, x) + '\n', mesh.faces))
    
    with open(fileName, "w") as fp:
        fp.writelines(v)
        fp.writelines(f)

