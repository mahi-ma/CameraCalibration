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

def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

pipeline = dai.Pipeline()

monoLeft = getMonoCamera(pipeline, 'LEFT')
monoRight = getMonoCamera(pipeline, 'RIGHT')

xoutLeft = pipeline.createXLinkOut()
xoutLeft.setStreamName("left")

xoutRight = pipeline.createXLinkOut()
xoutRight.setStreamName("right")

monoLeft.out.link(xoutLeft.input)
monoRight.out.link(xoutRight.input)

with dai.Device(pipeline) as device:
    leftQueue = device.getOutputQueue(name = "left", maxSize =1)
    rightQueue = device.getOutputQueue(name = "right", maxSize = 1)

    cv2.namedWindow("Stereo Pair")

    img_counter = 1
    while True:
        leftFrame = getFrame(leftQueue)
        rightFrame = getFrame(rightQueue)

        imOut = np.uint8(leftFrame/2 + rightFrame/2)

        cv2.imshow("Stereo Pair", imOut)

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

