import numpy as np
import cv2

imagesPaths1 = ["./stitching_images/secondSet/rialto1.jpeg","./stitching_images/secondSet/rialto2.jpeg","./stitching_images/secondSet/rialto3.jpeg"]
imagesPaths2 = ["./stitching_images/firstSet/studentcentereast1.jpg","./stitching_images/firstSet/studentcentereast2.jpg","./stitching_images/firstSet/studentcentereast3.jpg"]
imagesPaths3 = ["./stitching_images/thirdSet/bookstore1.jpg","./stitching_images/thirdSet/bookstore2.jpg","./stitching_images/thirdSet/bookstore3.jpg"]
imagesPaths4 = ["./stitching_images/fourthSet/bookstore5.jpg","./stitching_images/fourthSet/bookstore6.jpg","./stitching_images/fourthSet/bookstore7.jpg"]
imagesPaths5 = ["./stitching_images/fifthSet/tdeck1.jpg","./stitching_images/fifthSet/tdeck2.jpg","./stitching_images/fifthSet/tdeck3.jpg"]
images = []

for path in imagesPaths1:
	image = cv2.imread(path)
	images.append(image)

# stitcher=cv2.Stitcher.create()
print(len(images))
stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

if status == 0:
	# write the output stitched image to disk
	# cv2.imwrite("stichedOutput", stitched)
	# display the output stitched image to our screen
	print("stiching is successful")
	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)
# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
	print("[INFO] image stitching failed ({})".format(status))

