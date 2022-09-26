import numpy as np
import camera_matrix_calculation

#get camera matrix from caliberation program
cam_matx = camera_matrix_calculation.camera_matx

cam_matx_inv = np.linalg.inv(cam_matx)

#dimension of each cube on image capture_isp_2.png
#From x distance - 8 cms
# From y distance - 15.2 cms
# z is 28 cms
project_points = np.matrix([
    [8],
    [15.2],
    [28],
    [1]
])

dimension_matrix = cam_matx_inv.dot(project_points)
#dimension_matrix is real world coordinates of the object (churoco board in this case)
print(f"Length along x:{-1*dimension_matrix[0][0]}")
print(f"Length along y:{-1*dimension_matrix[1][0]}")

