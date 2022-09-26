import numpy as np
import camera_matrix_calculation

#get camera matrix from caliberation program
ie_matx = camera_matrix_calculation.camera_matx

ie_matx_inv = np.linalg.inv(ie_matx)

#dimension of each cube on image capture_isp_2.png
project_points = np.matrix([
    [8],
    [15.2],
    [28],
    [1]
])

dimension_matrix = ie_matx_inv.dot(project_points)
print(f"Length along x:{-1*dimension_matrix[0][0]}")
print(f"Length along y:{-1*dimension_matrix[1][0]}")

