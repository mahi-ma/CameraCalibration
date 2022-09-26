# CameraCalibration
Oak D Camera calibration using opencv

Answer 1.) Ran the code given on opencv documentation to calibrate oak d camera, passed images captured from the camera and calibrated the camera using opencv.calibrateCamera() and calculated the camera matrix which is intrinsic matrix * extrinsic matrix
Resultant Camera matrix= 
[[ 4.00394577e+02 -1.20082760e+01 -3.07589187e+02 -3.98792954e+03]
 [ 1.38419828e+01  4.09335485e+02 -2.48604813e+02 -4.92779226e+03]
 [ 1.37407248e-02 -6.48893950e-03 -9.99884536e-01 -1.32088778e+01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]


Answer 2.) Created program calculate_dimensions.py to calculate sides of our object using the perspective projection equation and calculated the length and width of our chess board.

Answer 3.) No its not feasible. I ran the RGB stream from mono camera using program depth_capture_mono.py and the depth_capture_stereo.py but got an error saying camera unrecognizable

Answer 4.) Ran the calibration script from luxinos documentation
Luxinos oak d camera calibration Intrinsic matrix:
[[448.824075   0.       316.351223 ]
 [  0.       448.824075 173.907494 ]
 [  0.         0.         1.       ]]


Our program Intrinsic matrix:
[[396.25760498  0.             313.0033623 ]
[ 0.            411.10607861   251.44362313]
[ 0.            0.             1.          ]]
