# CameraCalibration
Oak D Camera calibration using opencv

Answer 1.) Ran the code given on opencv documentation to calibrate oak d camera, passed images captured from the camera and calibrated the camera using opencv.calibrateCamera() and calculated the camera matrix which is intrinsic matrix * extrinsic matrix<br>
Resultant Camera matrix= <br>
[[ 4.00394577e+02 -1.20082760e+01 -3.07589187e+02 -3.98792954e+03]<br>
 [ 1.38419828e+01  4.09335485e+02 -2.48604813e+02 -4.92779226e+03]<br>
 [ 1.37407248e-02 -6.48893950e-03 -9.99884536e-01 -1.32088778e+01]<br>
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]<br>


Answer 2.) Created program calculate_dimensions.py to calculate sides of our object using the perspective projection equation and calculated the length and width of our chess board.

Answer 3.) Yes it is feasible to run both mono stream and stereo stream together. Oak-d device has 3 camera, 2 for mono stream, and middle one for rgb stream.

Answer 4.) Ran the calibration script from luxinos documentation<br>
Luxinos oak d camera calibration Intrinsic matrix:<br>
[[448.824075   0.       316.351223 ]<br>
[  0.       448.824075 173.907494 ]<br>
[  0.         0.         1.       ]]<br>


Our program Intrinsic matrix:<br>
[[396.25760498  0.             313.0033623 ]<br>
[ 0.            411.10607861   251.44362313]<br>
[ 0.            0.             1.          ]]<br>
