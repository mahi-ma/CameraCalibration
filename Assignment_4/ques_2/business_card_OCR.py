#Tasks I am doing:-
#QR code scanner -> send 
#edge detection/ Corner detetction
#details extractionn -> image segmentation (OCR)
#scan email and send link
#Face detection

import cv2;
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
import pytesseract
import numpy as np
import argparse
import imutils
import cv2
import re

bcard = cv2.imread("./business_card.jpeg")
print(bcard.shape)
plt.imshow(bcard,cmap="gray")
plt.show()

#details extraction on card using OCR

gray = cv2.cvtColor(bcard, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

plt.imshow(edged,cmap="gray")
plt.show()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not we are visualizing each step of the pipeline")
ap.add_argument("-c", "--min-conf", type=int, default=0,
	help="minimum confidence value to filter weak text detection")
args = vars(ap.parse_args())

orig = cv2.imread("./business_card.jpeg")
image = orig.copy()
image = imutils.resize(image, width=600)
ratio = orig.shape[1] / float(image.shape[1])


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# initialize a contour that corresponds to the business card outline
cardCnt = None

# loop over the contours
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		cardCnt = approx
		break

if cardCnt is None:
	raise Exception(("Could not find receipt outline. "
		"Try debugging your edge detection and contour steps."))


if args["debug"] > 0:
	output = image.copy()
	cv2.drawContours(output, [cardCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Business Card Outline", output)
	cv2.waitKey(0)

print(ratio)

#transform view
card = four_point_transform(orig, cardCnt.reshape(4, 2) * 0.0025)
# show transformed image
cv2.imshow("Business Card Transform", card)
cv2.waitKey(0)

# convert the business card from BGR to RGB channel ordering and then
# OCR it
rgb = cv2.cvtColor(bcard, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(rgb)
# use regular expressions to parse out phone numbers and email
# addresses from the business card
phoneNums = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)

nameExp = r"^[\w'\-,.][^0-9_!¡?÷?¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}"
names = re.findall(nameExp, text)

# show the phone numbers header
print("PHONE NUMBERS")
print("=============")
for num in phoneNums:
	print(num.strip())
# show the email addresses header
print("\n")
print("EMAILS")
print("======")
for email in emails:
	print(email.strip())
# show the name/job title header
print("\n")
print("NAME/JOB TITLE")
print("==============")
for name in names:
	print(name.strip())







