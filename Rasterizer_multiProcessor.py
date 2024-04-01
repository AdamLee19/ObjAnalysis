# import Mesh, MeshIO



import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from NumpyIO import writeFile, readFile
import cv2 as cv

import argparse
import time
import concurrent.futures
from math import ceil

class Rasterizer():
    def __init__(self, w: int = 400, h: int = 600) -> None:
        self.Width  = 400
        self.Height = 600



        self.img = np.zeros((self.Height, self.Width, 1), dtype = np.uint8)
        self.z_buffer = np.ones((self.Height, self.Width, 1), dtype = np.float16) * -np.inf 
    
        self.index = np.ones((self.Height, self.Width, 1), dtype = int) * -1

    def _isInside(self, x, p0, p1, p2) -> bool:
        vn = np.cross((p1 - p0), (p2 - p1))
        _2_A = np.linalg.norm(vn)
        n_hat = vn / _2_A
        is_inside = True
        u = np.dot(np.cross((p2 - p1), (x - p1)), n_hat) / _2_A
        if u < 0: is_inside = False
        
        v = np.dot(np.cross((p0 - p2), (x - p2)), n_hat) / _2_A
        if v < 0: is_inside = False

        if 1 - v - u < 0: is_inside = False

        return is_inside, u, v, 1 - v - u
    
    def _isInside_matrix(self, x, p0, p1, p2) -> NDArray:
        vn = np.cross((p1 - p0), (p2 - p1))
        _2_A = np.linalg.norm(vn)
        n_hat = vn / _2_A
        

        u = np.dot(np.cross((p2 - p1), (x - p1)), n_hat) / _2_A
        r = np.ones(u.shape, dtype=bool)
        r[u < 0] = False
        
        v = np.dot(np.cross((p0 - p2), (x - p2)), n_hat) / _2_A
        r[v < 0] = False
        
       
        r[1 - v - u < 0] = False
       
        return r
    
    def _valid_idx(self, r, c):
        if r < 0:
            return False
        if c < 0:
            return False
        if r >= self.Height:
            return False
        if c >= self.Width:
            return False
        return  True 

    def _rasterize_tri(self, triangle: NDArray, index: NDArray, ignore: NDArray):
        for i in range(3):
            if int(index[i]) in ignore:
                return

        p0 = triangle[0, :3]
        p1 = triangle[1, :3]
        p2 = triangle[2, :3]
        
        
        
       
        P = np.vstack((p0, p1, p2))


        min_ = np.clip(P[:, :2].min(axis=0), 0, [self.Width - 1, self.Height - 1]).astype(int)
        max_ = np.clip(P[:, :2].max(axis=0), 0, [self.Width - 1, self.Height - 1]).astype(int)
        
        tri = []
        for i in range(3):
            c, r = triangle[i, :2].astype(int)
            z = triangle[i, 2]
            if self._valid_idx(r, c) and z > self.z_buffer[r, c] :
                self.img[r, c] = 255
                self.z_buffer[r, c] = z
                self.index[r, c] = int(index[i])
            tri.append((c, r))
       

        for r in range(min_[1], max_[1] + 1):
            for c in range(min_[0], max_[0] + 1):
                if (c, r) in tri: continue
                b, u, v, w = self._isInside(np.array([c, r]), p0[:2], p1[:2], p2[:2])
                if b:
                    x = u * p0 + v * p1 + w * p2
                    if x[2] > self.z_buffer[r, c]:
                        self.img[r, c] = x[2] * 100
                        self.z_buffer[r, c] = x[2]
                        self.index[r, c] = -1
        
        # grid1, grid2 = np.meshgrid(np.arange(min_[1], max_[1] + 1), np.arange(min_[0], max_[0] + 1), indexing='ij')
        # result = np.stack((grid2.ravel(), grid1.ravel()), axis=-1)
        # grid = self._isInside_matrix(result, p0, p1, p2)

       
        # c = result[grid == True][:, 0]
        # r = result[grid == True][:, 1] 
        # self.img[r, c] = 255
       
         
    def _rasterize_quad(self, quad: NDArray, index: NDArray, ignore: NDArray) -> None:
        self._rasterize_tri(quad[:3], index[:3], ignore)
        self._rasterize_tri(quad[3:], index[3:], ignore)
        
        

    # def rasterize(self, points: NDArray, facesIdx: NDArray, ignore: NDArray) -> None:
       
    #     faces = np.array(list(map(lambda fidx: points[fidx], facesIdx)))
    #     indices = np.array(list(map(lambda fidx: fidx, facesIdx)))
        
    #     #bsize = faces.shape[0]
    #     #faces = faces.reshape(bsize, -1)
    #     #np.apply_along_axis(self._rasterize_quad, 1, faces)
    #     for i, f in enumerate(faces):
    #         self._rasterize_quad(f, indices[i])

    def rasterize(self, points: NDArray, facesIdx: NDArray, ignore: NDArray) -> None:
       
        faces = np.array(list(map(lambda fidx: points[fidx], facesIdx)))
        indices = np.array(list(map(lambda fidx: fidx, facesIdx)))
        
        
        #bsize = faces.shape[0]
        #faces = faces.reshape(bsize, -1)
        #np.apply_along_axis(self._rasterize_quad, 1, faces)
        for i, f in enumerate(faces):
            self._rasterize_quad(f, indices[i], ignore) 





    def _write_visible_index(self, fileName): 
        idx = self.index[self.index != -1]
       
        idx = np.array(list(set(idx.tolist())))
        writeFile(fileName, idx)
        
       

        

    def save_img(self, name: str) -> None:
        
        cv.imwrite(name, self.img)


def main(camera: str, start: int, end: int) -> None:

    print(camera, start, end, end='\n\n\n')
    dataFolder = Path('./10000')
    viewMatDir   = Path(f'./common/Camera/{camera}/view.npy')
    projMatDir   = Path(f'./common/Camera/{camera}/project.npy')
    normalDir   = Path('./common/VertexNormal/Normal.npy')
    eigenVecDir = Path('./common/Point/EigenVector.npy')
    meanDir = Path('./common/Point/Mean.npy')
    faceDir = Path('./common/FaceIndex/Faces.npy')

    for i in range(start, end):
        imgRes = (400, 600)
        rasterizer = Rasterizer(imgRes[0], imgRes[1])
        faceName = Path(f'Face{i:0>{5}}')
        dir = dataFolder / faceName
        

        modelMatDir = dir / Path('ModelMat.npy')
        paramDir = dir / Path('PointParam.npy')

        modelMat = readFile(modelMatDir)
        viewMat = readFile(viewMatDir)
        projMat = readFile(projMatDir)
        normal = readFile(normalDir)
        param = readFile(paramDir)
        mean = readFile(meanDir)
        eigenVec = readFile(eigenVecDir)
    
        faces = readFile(faceDir)
    

        obj = (eigenVec @ param + mean).reshape(-1, 3)
        numPoints = obj.shape[0]
    
        ones = np.ones((numPoints, 1))
        obj = np.hstack((obj, ones)).T
        zeros = np.zeros((numPoints, 1))
        normal = np.hstack((normal, zeros)).T 
        

        pvmMat = projMat @ viewMat @ modelMat
        obj_pvm = (pvmMat @ obj).T
        third = obj_pvm[:,2].reshape(-1, 1)
        fourth = obj_pvm[:, 3].reshape(-1, 1)
        result = (obj_pvm / fourth / third)
        result[:, 0] = (1 + result[:, 0]) * imgRes[0] * 0.5
        result[:, 1] = (1 - result[:, 1]) * imgRes[1] * 0.5

        vmMat = viewMat @ modelMat
        view_direction = -(vmMat @ obj)[:3, :].T
        view_direction = view_direction / np.linalg.norm(view_direction, axis=1)[:, np.newaxis]
        normal = (vmMat @ normal)[:3, :].T
        normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
        cos = np.sum(view_direction * normal, axis=1)

        ignore = np.where(cos<=0)

        s = time.time()
        rasterizer.rasterize(result, faces, ignore[0])
        fileName = dir / Path(f'{camera}_visible.npy')
        rasterizer._write_visible_index(fileName)
        print(f'File: {dir.name}-{time.time() - s}s')
        rasterizer.save_img(str(dir / Path(f"{camera}.jpeg")))   

       

if __name__ == "__main__":
    from threading import Thread
    

    parser = argparse.ArgumentParser(description='Visibility')
    parser.add_argument('--camera', type=str, help="left, front, right?", required=True)
    parser.add_argument('--start', type=int, required=True, help="Start face")
    parser.add_argument('--end',  type=int, required=True, help="End face")
    args = parser.parse_args()
    
    #main('left', 0, 2)
    import os
    num_threads =os.cpu_count()
    print(num_threads)

    start = args.start
    end = args.end

    interval = ceil((end - start) / num_threads)
    
    pairs = []
    for i in range(num_threads):
        s = i * interval + start
        e = s + interval if (s + interval) <= end else end 
        
        pairs.append((s, e))    

    #main('left', args.start, args.end)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(main, args.camera, s, e) for s, e in range(pairs)]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    
    



       

       

    

    

    

    
    
    
    
    
    
   
   
    