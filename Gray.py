from pathlib import Path
from NumpyIO import *
import cv2 as cv
import numpy as np

eigenVecDir = Path('./common/Point/EigenVector.npy')
eigen_vec = readFile(eigenVecDir)
print(eigen_vec.shape)
meanDir = Path('./common/Point/Mean.npy')
mean_vec = readFile(meanDir)
print(mean_vec.shape)




print('\n\n\n\n')





view = ["front", "left", "right"]
dataFolder = Path('./5000')	
		
			


for d in dataFolder.iterdir():
	if not d.is_dir(): continue
	
	folderName = d.name
	
	print(f"File: {folderName}")
	for v in view:
		img = cv.imread(str(d / Path(f'{v}ViewShape.jpg')), cv.IMREAD_GRAYSCALE)
		print(img.shape)
		cv.imwrite(str(d / Path(f'{v}ViewShape_g.jpg')), img) 

		
		
	












		