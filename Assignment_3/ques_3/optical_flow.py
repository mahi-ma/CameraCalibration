import numpy as np
import cv2



def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


# i) keeping preevious frame as reference frame
cap = cv2.VideoCapture('./room-video.webm')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

n = 32
count = 0
frameNo = 0
frame11th = 0
frame31st = 0

#keeping every previous frame as reference frame
while suc:

    suc, img = cap.read()
    frameNo = frameNo + 1
    if frameNo==11:
        frame11th = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if frameNo==31:
        frame31st = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if(suc and count%n == 0):
        print(count)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        cv2.imshow('flow', draw_flow(gray, flow))
    
    count += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break



# ii) keeping every 11th frame as reference frame
cap = cv2.VideoCapture('./room-video.webm')

suc, prev = cap.read()
frameNo = 0

while suc:

    suc, img = cap.read()
    frameNo = frameNo + 1
    if frameNo==11:
        frame11th = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frameNo=0
    if(suc and count%n == 0):
        print(count)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(frame11th, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        cv2.imshow('flow', draw_flow(gray, flow))
    
    count += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



# iii) keeping every 31s frame as reference frame
cap = cv2.VideoCapture('./room-video.webm')

suc, prev = cap.read()
frameNo = 0

while suc:

    suc, img = cap.read()
    frameNo = frameNo + 1
    if frameNo==31:
        frame31st = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frameNo=0
    if(suc and count%n == 0):
        print(count)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(frame31st, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        cv2.imshow('flow', draw_flow(gray, flow))
    
    count += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()