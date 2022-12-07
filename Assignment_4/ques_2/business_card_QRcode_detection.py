import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

# Display barcode and QR code location
def display(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[ (j+1) % n][0]), (255,0,0), 3)
 
    # Display results
    cv2.imshow("Results", im)

bcard = cv2.imread("./business_card.jpeg")
print(bcard.shape)
plt.imshow(bcard,cmap="gray")
plt.show()

qrDecoder = cv2.QRCodeDetector()
value, points, straight_qrcode = qrDecoder.detectAndDecode(bcard)
print(value)
 
# Detect and decode the qrcode
data,bbox,rectifiedImage = qrDecoder.detectAndDecode(bcard)
if len(data)>0:
    print("Decoded Data : {}".format(data))
    display(bcard, bbox)
    rectifiedImage = np.uint8(rectifiedImage);
    cv2.imshow("Rectified QRCode", rectifiedImage);
else:
    print("QR Code not detected")
    cv2.imshow("Results", bcard)
 
cv2.waitKey(0)
cv2.destroyAllWindows()