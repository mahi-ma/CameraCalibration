import numpy as np
import part1_calibrate_compare

# ie_matx = np.matrix([
#     [634.746364, 0, 499.508324, 0],
#     [0, 634.746364, -305.107872, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ])

ie_matx = part1_calibrate_compare.camera_matx

ie_matx_inv = np.linalg.inv(ie_matx)

project_points = np.matrix([
    [8],
    [15.2],
    [28],
    [1]
])

# row_vector = np.matrix([1,2,3,4])
# print(row_vector)

print(ie_matx_inv.dot(project_points))
