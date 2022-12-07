#SIFT feature extraction

import cv2;
import matplotlib.pyplot as plt

bcard = cv2.imread("./business_card.jpeg")
plt.imshow(bcard,cmap="gray")
plt.show()

grayImg = cv2.cvtColor(bcard,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(grayImg,None)

print(kp1,des1)
plt.imshow(des1)
plt.show()