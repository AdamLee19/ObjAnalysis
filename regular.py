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
        


 

    def get_data(self, param):
        return self.mean + self.eigen_vectors @ param

    
    

pca = Path('./Combine/Diffuse')



diffuse = PCA(pca)


for n in Path('./npys/').glob('*.npy'):
    

    c = readFile(n).reshape(-1, 1)
    print(n, c.shape)
    print(c) 
    face = n.stem


    x = diffuse.get_data(c)
    
    img = x.reshape((1024, 1024, 3)) * 255.0
    img = img.astype('uint8')       
    img_pil = Image.fromarray(img)
    img_pil.save(n.parent.joinpath(f"{face}_diffuse.jpg"))
  

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

    

