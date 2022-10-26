import cv2
import numpy as np
import matplotlib.pyplot as plt

src_points = np.array([[2009,1529],[2025,1548],[2025,2015],[1868,983],[2344,959]])
dest_points = np.array([[2696,1488],[2714,1508],[2726,1992],[2542,931],[3064,868]])

# src_points = np.array([[1358,2391],[1386,2391],[1351,2423],[1385,2424],[1650,1872]])
# dest_points = np.array([[1356,2393],[1387,2393],[1351,2426],[1385,2427],[1650,1874]])


h, status = cv2.findHomography(src_points, dest_points)

im_src = cv2.imread('capture_isp_3.png')
im_dst = cv2.imread('capture_isp_2.png')

#-> this will give warped image output using h as homography matrix
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

cv2.imshow("Warped_Source_Image", im_out)
plt.imshow(im_out)
plt.show()

print(h)

#homography matrix
#[[ 1.68719644e-01  2.60876523e-02  1.34357960e+03]
# [-2.12792802e-01  6.65053711e-01  3.60725030e+02]
# [-1.82383008e-04  3.38642018e-06  1.00000000e+00]]


