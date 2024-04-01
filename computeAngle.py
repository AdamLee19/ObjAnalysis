import numpy as np
from math import asin, pi, atan2, cos, sin

# M = np.array([0.99651642, -0.05945409, 0.05848272, 
# -0.08282197, -0.78771701, 0.61044445, 
# 0.00977441, -0.61316158, -0.78989705]).reshape(3,3)

M = np.load('rotationMatrix.npy')
print(M)
#M[0, :] = -M[0, :]
M[1, :] = -M[1, :]
M[:, 2] = -M[:, 2]
# M = -M
print(np.linalg.det(M))


print()
R11 = M[0, 0]
R12 = M[0, 1]
R13 = M[0, 2]
R21 = M[1, 0]
R22 = M[1, 1]
R23 = M[1, 2]
R31 = M[2, 0]
R32 = M[2, 1]
R33 = M[2, 2]
print(M)
print(R11, R12, R13)
print(R21, R22, R23)
print(R31, R32, R33)


theta1 = -asin(R31)
theta2 = pi - theta1

psi1 = atan2((R32 / cos(theta1)),  (R33 / cos(theta1)))
psi2 = atan2((R32 / cos(theta2)),  (R33 / cos(theta2)))



phi1 = atan2((R21 / cos(theta1)),  (R11 / cos(theta1)))
phi2 = atan2((R21 / cos(theta2)),  (R11 / cos(theta2)))
print(psi1 * 180 / pi, theta1 * 180 / pi,  phi1 * 180 / pi)
print(psi2 * 180 / pi, theta2 * 180 / pi,  phi2 * 180 / pi)




Rx = np.array([1, 0 ,0, 0, cos(psi1), -sin(psi1), 0, sin(psi1), cos(psi1)]).reshape(3,3)
Ry = np.array([cos(theta1), 0 , sin(theta1), 0, 1, 0, -sin(theta1), 0,cos(theta1)]).reshape(3,3)
Rz = np.array([cos(phi1), -sin(phi1) ,0, sin(phi1), cos(phi1), 0, 0, 0, 1]).reshape(3,3)

print(Rz @ Ry @ Rx)






