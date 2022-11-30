import numpy as np
import cv2
import matplotlib.pyplot as plt

captured_image = cv2.imread("capture_isp_3.png")
edge_image = cv2.Canny(captured_image,180,200)

print(captured_image)
plt.imshow(captured_image)
plt.imshow(edge_image,cmap="gray")
plt.show()