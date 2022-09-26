import numpy as np
import cv2 as cv
import glob
from scipy.spatial.transform import Rotation
from math import cos, sin, radians

def trig(angle):
    r = radians(angle)
    return cos(r), sin(r)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('chess\leftImages\*.png')
print(images)
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# objpoints1 = np.array(objpoints)
# objpoints1 = objpoints1.astype('float64')
# imgpoints1 = np.array(imgpoints)
# imgpoints1 = imgpoints1.astype('float64')

# flag = cv.SOLVEPNP_ITERATIVE
# retval, rvecs, tvecs = cv.solvePnP(objpoints1, imgpoints1, mtx, dist, flags= 0)

# print("rotation matrix:")
# print(cv.Rodrigues(np.array(rvecs)))
r_obj = Rotation.from_rotvec(np.array(rvecs[0]).reshape(1,3))
rot_matrix = r_obj.as_matrix()

nprvecs = np.array(rvecs)
nptvecs = np.array(tvecs)

xc, xs = trig(nprvecs[0][0][0])
yc, ys = trig(nprvecs[0][1][0])
zc, zs = trig(nprvecs[0][2][0])

tx = nptvecs[0][0][0]
ty = nptvecs[0][1][0]
tz = nptvecs[0][2][0]

translation_mtx = np.array([
    [1,0,0,tx],
    [0,1,0,ty],
    [0,0,1,tz],
    [0,0,0,1]
])

rotation_x_matx = np.array([
    [1,0,0,0],
    [0,xc,-xs,0],
    [0,xs,-xc,0],
    [0,0,0,1]
])

rotation_y_matx = np.array([
    [yc,0,ys,0],
    [0,1,0,0],
    [-ys,0,yc,0],
    [0,0,0,1]
])

rotation_z_matx = np.array([
    [zc,-zs,0,0],
    [zs,zc,0,0],
    [0,0,1,0],
    [0,0,0,1]
])

extrensic_matx = np.dot(rotation_z_matx, np.dot(rotation_y_matx, np.dot(rotation_x_matx, translation_mtx)))
intrinsic_matx = np.append( np.append(mtx, [[0],[0],[1]], axis=1), [np.array([0,0,0,1])], axis=0)
camera_matx = np.dot(intrinsic_matx, extrensic_matx)
print(camera_matx)

cv.destroyAllWindows()