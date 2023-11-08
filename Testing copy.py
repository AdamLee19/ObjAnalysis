from pathlib import Path
from NumpyIO import *
import MeshIO as mio
import Mesh
import numpy as np
from PIL import Image



class PCA(object):
    
    def __init__(self, folder, data_dir = Path('.')):

        self.eigen_vectors = readFile(data_dir.joinpath(folder, 'EigenVector.npy'))
        self.eigen_values = readFile(data_dir.joinpath(folder, 'EigenValue.npy')).reshape(-1, 1)
        self.mean = readFile(data_dir.joinpath(folder, 'Mean.npy'))
        
        self.param =  self.eigen_vectors.T @ (self.mean - self.mean) 

        print(folder)
        print(f'\tEigen Vector: {self.eigen_vectors.shape}')
        print(f'\tEigen Value: {self.eigen_values.shape}')
        print(f'\tMean: {self.mean.shape}')
        

    def sum_variance(self):
        return self.eigen_values.sum()

    def perturb_param(self):
        
        sDeviation = np.sqrt(self.eigen_values)
        self.param = np.clip(np.random.normal(scale = sDeviation), -sDeviation * 3.0, sDeviation * 3.0)

   
       

    def get_param(self):
        return self.param

    def get_mean(self):
        return self.mean

    def get_data(self):
        return self.mean + self.eigen_vectors @ self.param

class CombinePCA(PCA):
    def __init__(self, folder, data_dir = Path('.')):
        super().__init__(folder, data_dir)
        
        self.comb_vectors = readFile(data_dir.joinpath(folder, 'CombVector.npy'))
        self.comb_value = readFile(data_dir.joinpath(folder, 'CombValue.npy')).reshape(-1, 1)
        self.weight = readFile(data_dir.joinpath(folder, 'Weight.npy'))

        self.compute_combine_param()

        print(f'\tCombine Eigen Vector: {self.comb_vectors.shape}')
        print(f'\tCombine Eigen Value: {self.comb_value.shape}')
        print(f'\tCombine Weight: {self.weight.shape}')
    
    def perturb_base_param(self):
        self.perturb_param()
        self.compute_combine_param()


    def compute_combine_param(self):
        self.combine_param = self.comb_vectors.T @ self.weight @ self.param
        return self.combine_param


    def update_param_by_c(self, c):
        self.combine_param = c
        w_inv = np.linalg.inv(self.weight)
        self.param = w_inv @ self.comb_vectors @ c


    def weightMatrix_identity(self):
        rows, _ = self.eigen_vectors.T.shape
        return np.identity(rows)

    def weightMatrix_variance(self, vari_other):
        rows, _ = self.eigen_vectors.T.shape
        lambda_s = self.eigen_values.sum()
        lambda_o = vari_other.sum()
        r = lambda_o / lambda_s
        weights = np.zeros((rows, rows))
        np.fill_diagonal(weights, r)
        return weights
    
    

combine = Path('./Combine')


points = CombinePCA('Points', combine)
diffuse = CombinePCA('Diffuse', combine)
normal = CombinePCA('Normal', combine)
gloss = CombinePCA('Gloss', combine)

for n in Path('./npys/').glob('*.npy'):
    

    c = readFile(n).reshape(-1, 1)
    print(n, c.shape)
    print(c) 
    face = n.stem

    points.update_param_by_c(c)
    x = points.get_data()
    objDir = Path('./mean.obj')
    mesh = Mesh.Mesh()
    mio.readMesh(mesh, objDir)
    mesh.updateVert(x.reshape(-1,3).tolist())
    mio.writeMesh(mesh,  n.parent.joinpath(f"{face}.obj"))
    
    # diffuse.update_param_by_c(c)
    # changed = diffuse.get_data()
    # img = changed.reshape((1024, 1024, 3)) * 255.0
    # img = img.astype('uint8')       
    # img_pil = Image.fromarray(img)
    # img_pil.save(n.parent.joinpath(f"{face}_diffuse.jpg"))
  

# c = CEVector_d.T @ b_d
# # c = CEVector_d.T @ (EVector_d.T @ (mean_d - mean_d))
# # c[0][0] = np.sqrt(CEValue_d).flatten()[0]*30
# print(c[0][0])
# changed = mean_d + Q_d @ c


    # normal.update_param_by_c(c)
    # changed = normal.get_data()
    # img = changed.reshape((1024, 1024, 3)) * 255.0
    # img = img.astype('uint8')       
    # img_pil = Image.fromarray(img)
    # img_pil.save('normal.jpg')

    # gloss.update_param_by_c(c)
    # changed = gloss.get_data()
    # img = changed.reshape((1024, 1024, 3)) * 255.0
    # img = img.astype('uint8')       
    # img_pil = Image.fromarray(img)
    # img_pil.save('gloss.jpg')

    

