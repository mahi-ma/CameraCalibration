from operator import imod
from unicodedata import name
import cv2
import depthai as dai
import numpy as np

def getMonoCamera(pipeline, side):
    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    if side == 'LEFT':
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    return mono

def getStereoPair(pipeline, monoLeft, monoRight):
    stereo = pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo

def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

pipeline = dai.Pipeline()

monoLeft = getMonoCamera(pipeline, 'LEFT')
monoRight = getMonoCamera(pipeline, 'RIGHT')

stereo_pair = getStereoPair(pipeline, monoLeft, monoRight)

xout_display = pipeline.createXLinkOut()
xout_display.setStreamName("Disparity")

xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("leftrectify")
xout_right = pipeline.createXLinkOut()
xout_right.setStreamName("rightrectify")

stereo_pair.disparity.link(xout_display.input)

stereo_pair.rectifiedLeft.link(xout_left.input)
stereo_pair.rectifiedRight.link(xout_right.input)

img_counter = 0

with dai.Device(pipeline) as device:
    disparity_queue = device.getOutputQueue(name = "Disparity", maxSize = 1)
    left_queue = device.getOutputQueue(name = "leftrectify", maxSize =1)
    right_queue = device.getOutputQueue(name = "rightrectify", maxSize =1)

    multiplier = 255/stereo_pair.getMaxDisparity()

    cv2.namedWindow("Stereo")

    while True:
        disparity = getFrame(disparity_queue)
        disparity = (disparity*multiplier).astype(np.uint8)

        left_frame = getFrame(left_queue)
        right_frame = getFrame(right_queue)

        imOut = np.uint8((left_frame + right_frame)/2)

        imOut = cv2.cvtColor(imOut, cv2.COLOR_GRAY2RGB)

        cv2.imshow("Stereo", imOut)
        cv2.imshow("Disparity", disparity)

        k = cv2.waitKey(1)
        if k%256 ==27:
            #Esc pressed
            print("Escape hit, closing operation")
            break
        elif k%256 == 32:
            #Space pressed
            img_name = f"opencv_picture_{img_counter}.png"
            cv2.imwrite(img_name, imOut)
            print(f"{img_name} saved")
            img_counter += 1

cv2.destroyAllWindows()
