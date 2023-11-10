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


fp = open('./common/ibug68.txt', 'r')
a = fp.readlines()
fp.close()
lm_idx = [int(l) for l in a]
print(len(lm_idx))

print('\n\n\n\n')



cameraList = ['left', 'front', 'right']
dataFolder = Path('./5000')
imgRes = (400, 600)


	
		
			

for c in cameraList:
	viewMat = readFile('common/Camera/' + c + '/view.npy')
	projMat = readFile('common/Camera/' + c + '/project.npy')

	mat = projMat @ viewMat
	for d in dataFolder.iterdir():
		if not d.is_dir(): continue
		
		
			
		folderName = d.name
		
		print(f"File: {folderName} - Camera: {c}")
		
		paramDir = d / Path('PointParam.npy')
		face_vec = readFile(paramDir).reshape(-1, 1)

		obj = (eigen_vec @ face_vec + mean_vec).reshape(-1, 3)

		lmks = obj[lm_idx, :]
		ones = np.ones((68, 1))
		lmks = np.hstack((lmks, ones))
		

		modelMat = readFile(d / Path('ModelMat.npy'))
		


		cam_matrix = mat @ modelMat	

		result = (cam_matrix @ lmks.T).T
		fourth = result[:, 3].reshape(-1, 1) 
		third = result[:, 2].reshape(-1, 1) 
		result = result / fourth
		result = result / third
		
		

		result[:, 0] = (1 + result[:, 0]) * imgRes[0] * 0.5
		result[:, 1] = (1 - result[:, 1]) * imgRes[1] * 0.5
		result = result.astype('int')
		img = cv.imread(str(d / Path(c+'ViewShape.jpg')))
		for p in result:
			cv.circle(img, (p[0], p[1]), 2, (0, 0, 255), -1)
		cv.imshow(c, img)
		cv.waitKey(0)
		cv.destroyAllWindows()

		
		
	












		