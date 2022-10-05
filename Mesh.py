import array
from numpy import cross, amax, amin, vstack
from numpy.linalg import norm
from numpy import array as nparray

class Mesh:
    def __init__(self, name = '', v = None, f = None, n = None, vt = None):
        if v == None: v = []
        if f == None: f = []
        if n == None: n = []
        if vt == None: vt = []

        self.name = name

        self.vertices = v #for the 48 model, it has 12466 vertices
        self.normal = n
        self.texCoord = vt
        self.faces = f

        
        self.point_per_face = 0 if len(f) == 0 else len(f[0])
        self.n_vertex = len(self.vertices)

        self.bbox = []
        self.center = []
        self._compute_bbox()
        
    
    def updateVert(self, v):
        self.vertices = v
        self._compute_bbox()
        
    def get_texCoord(self):
        texCoord = array.array('f', [])
        for f in self.faces:
            f = f.copy()
            if self.point_per_face == 4:
                f.insert(2, f[2])
                f.append(f[0])
            index = list(map(lambda x: int(x.split('/')[1]) - 1, f))
            vt = [i for sublist in index for i in self.texCoord[sublist]]
            texCoord.extend(vt)
        return texCoord
        
    def get_vertex(self):
        vertices = array.array('f',[])
        for i in range(len(self.faces)):
            face = self.faces[i].copy()
            if self.point_per_face == 4:
                face.insert(2, face[2])
                face.append(face[0])
            index = list(map(lambda x: int(x.split('/')[0]) - 1, face))
            
            points = [i for sublist in index for i in self.vertices[sublist]]
            vertices.extend(points)
        return vertices

    def get_normal(self):
        normals = array.array('f',[]) 

        if len(self.normal) == 0:
            return self._get_normal()
        
        print('Fix here')
        return normals


    def _get_normal(self):
        normals = array.array('f',[])
        for i in range(len(self.faces)):
            face = self.faces[i].copy()
            if self.point_per_face == 4:
                face.insert(2, face[2])
                face.append(face[0])
                index = list(map(lambda x: int(x.split('/')[0]) - 1, face))
                verts = list(map(lambda v: self.vertices[v], index))
                n1 = self._compute_norm(verts[0], verts[1], verts[2])
                n2 = self._compute_norm(verts[3], verts[4], verts[5])
                normals.extend(n1 * 3)
                normals.extend(n2 * 3)
            elif self.point_per_face == 3:
                pass
        return normals
        
        
        
    def _compute_norm(self, v0, v1, v2):
        p0 = nparray(v0)
        p1 = nparray(v1)
        p2 = nparray(v2)
        
        a1 = p0 - p2
        a2 = p1 - p2
        n = cross(a1, a2)
        n /= norm(n)
        return array.array('f', n)

    def _compute_bbox(self):
        if(len(self.vertices) != 0):
            vertMat = nparray(self.vertices)
            Max = amax(vertMat, axis = 0)
            Min = amin(vertMat, axis = 0)
            #xMin, xMax, yMin, yMax, zMin, zMax
            bbox = vstack((Min, Max)).T.flatten()
            center = (Max + Min) / 2.0
            self.bbox = list(bbox)
            self.center = list(center)
            
    def normalizeCoord(self):
        point = nparray(self.vertices)
        
        bboxMin = nparray([self.bbox[0], self.bbox[2], self.bbox[4]]).reshape(1, 3)
        bboxMax = nparray([self.bbox[1], self.bbox[3], self.bbox[5]]).reshape(1, 3)
    
    

        point = 2 * (point - bboxMin) / (bboxMax - bboxMin) - 1

        self.vertices = point.tolist()
        self._compute_bbox()
            
        

