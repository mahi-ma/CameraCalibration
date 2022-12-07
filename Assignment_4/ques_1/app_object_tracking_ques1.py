# Steps :-
# 1.) Object recognition
# 2.) Motion tracking of considered object
# 3.) Dimention calculation of object -> real time

import cv2
import numpy as np
import numpy
from matplotlib import pyplot as plt
import tensorflow
import camera_matrix_calculation
import numpy as np
import numba as nb
import depthai as dai

# sampleImg = cv2.imread("./objectImages/found/capture_isp_1.png")
# print(sampleImg.shape)

#train model on images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator()

gen = idg.flow_from_directory("./imgs",target_size=(200,200))

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Dropout

inlayer = Input(shape=(200,200,3))
c1 = Conv2D(16, 3, activation="relu")(inlayer)
mp1 = MaxPooling2D(2)(c1)
c2 = Conv2D(32, 3, activation="relu")(mp1)
mp2 = MaxPooling2D(2)(c2)
flat = Flatten()(mp2)
d2 = Dense(300, activation="relu")(flat)
d3 = Dense(100, activation="relu")(d2)
drop1 = Dropout(.2)(d3)
d5 = Dense(50, activation="relu")(d3)
out_layer = Dense(2, activation="softmax")(d5)

model = Model(inlayer, out_layer)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(gen, epochs=12)

streams = []
# Enable one or both streams
streams.append('isp')

''' Packing scheme for RAW10 - MIPI CSI-2
- 4 pixels: p0[9:0], p1[9:0], p2[9:0], p3[9:0]
- stored on 5 bytes (byte0..4) as:
| byte0[7:0] | byte1[7:0] | byte2[7:0] | byte3[7:0] |          byte4[7:0]             |
|    p0[9:2] |    p1[9:2] |    p2[9:2] |    p3[9:2] | p3[1:0],p2[1:0],p1[1:0],p0[1:0] |
'''
# Optimized with 'numba' as otherwise would be extremely slow (55 seconds per frame!)
@nb.njit(nb.uint16[::1] (nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0

   #for i in np.arange(input.size // 5): # around 25ms per frame (with numba)
    for i in nb.prange(input.size // 5): # around  5ms per frame
        b4 = input[i * 5 + 4]
        out[i * 4]     = ((input[i * 5]     << 2) | ( b4       & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) |  (b4 >> 6)       ) << lShift

    return out

print("depthai version:", dai.__version__)
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

if 'isp' in streams:
    xout_isp = pipeline.createXLinkOut()
    xout_isp.setStreamName('isp')
    cam.isp.link(xout_isp.input)

device = dai.Device(pipeline)
device.startPipeline()

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=True)
    q_list.append(q)
    # Make window resizable, and configure initial size
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (960, 540))



def getClassName(index):
    if index == 0:
        return "object found"
    else:
        return "object not found"

font = cv2.FONT_HERSHEY_COMPLEX
# org
org = (50, 50)
size = ""

#calculate dimensions
foundImg = cv2.imread("./objectImages/found/capture_isp_1.png")
res,coords = model.predict(np.array([foundImg]))

(fX, fY, fW, fH) = coords

size = "(" + str(fW) + "," + str(fH) + ")"

# fontScale
fontScale = 8
# Blue color in BGR
color = (255, 0, 0)  
# Line thickness of 2 px
thickness = 2

def img_alignment(img1, img2):
    img1, img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
    img_size = img1.shape
    warp_mode = cv2.MOTION_TRANSLATION

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3,dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3,dtype=np.float32)
    
    n_iterations = 5000
    termination_eps = 1e-10

    criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, n_iterations, termination_eps)

    cc, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(img2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img2_aligned = cv2.warpAffine(img2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return img2_aligned

#capture video stream using stereo camera
capture_flag = False
img_counter = 0
while True:
    for index,q in enumerate(q_list):
        frame1name = q.getName()
        frame1data = q.get()
        if index+1 < len(q_list):
            frame2name = q_list[index+1].getName()
        else:
            frame2name = q_list[index].getName()
        
        if index+1 < len(q_list):
            frame2data = q_list[index+1].get()
        else:
            frame2data = q_list[index].get()
        width, height = frame1data.getWidth(), frame1data.getHeight()
        width1,height1 = frame2data.getWidth(),frame2data.getHeight()

        payload = frame1data.getData()
        payload1 = frame2data.getData()
        # capture_file_info_str = ('capture_' + name
        #                         #  + '_' + str(width) + 'x' + str(height)
        #                          + '_' + str(data.getSequenceNum())
        #                         )
        capture_file_info_str = f"capture_{frame1name}_{img_counter}"
        capture_file_info_str1 = f"capture_{frame1name}_{img_counter + 1}"
        if frame1name == 'isp':
            shape = (height * 3 // 2, width)
            yuv420p = payload.reshape(shape).astype(np.uint8)
            yuv420p1 = payload1.reshape(shape).astype(np.uint8)
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
            bgr1 = cv2.cvtColor(yuv420p1, cv2.COLOR_YUV2BGR_IYUV)
            grayscale_img =  cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            grayscale_img2 = cv2.cvtColor(bgr1,cv2.COLOR_BGR2GRAY)
        if capture_flag:  # Save to disk if 'space' was pressed
            filename = capture_file_info_str + '.png'
            print("Saving to file:", filename)
            grayscale_img = np.ascontiguousarray(grayscale_img)  # just in case
            img2 = np.ascontiguousarray(img2) 
            cv2.imwrite(filename, grayscale_img)
        bgr = np.ascontiguousarray(bgr)  # just in case
        res = model.predict(np.array([cv2.resize(bgr, (200,200))]))
        bgr = cv2.putText(bgr, getClassName(res.argmax(axis=1)) , org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        if getClassName(res.argmax(axis=1))=="object found":
            bgr = cv2.putText(bgr,size,(100,100), font,fontScale, color, thickness, cv2.LINE_AA)

        diff = cv2.absdiff(bgr, bgr1)
        
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        diff_blur = cv2.GaussianBlur(diff_gray, (5,5,), 0)

        _, binary_img = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, b, l = cv2. boundingRect(contour)
            if cv2.contourArea(contour) > 300:
                cv2.rectangle(bgr, (x, y), (x+b, y+l), (0,255,0), 2)
        cv2.imshow(frame1name, bgr)
    # Reset capture_flag after iterating through all streams
    capture_flag = False
    key = cv2.waitKey(5)
    if key%256 == 27:
        #escape 
        print("Operation over")
        break
    elif key%256 == 32:
        #space to click picture
        capture_flag = True
        img_counter += 1


# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# obj_data = cv2.CascadeClassifier('./data.xml')

# found = obj_data.detectMultiScale(img_gray, minSize =(20, 20))

# amount_found = len(found)

# if amount_found != 0:
      
#     for (x, y, width, height) in found:

#         cv2.rectangle(img_rgb, (x, y), 
#                       (x + height, y + width), 
#                       (0, 255, 0), 5)
          

# plt.subplot(1, 1, 1)
# plt.imshow(img_rgb)
# plt.show()