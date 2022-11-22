import cv2
import matplotlib.pyplot as plt

interest_img = cv2.imread("../ques1Images/4.png")

cropped_img = interest_img[714:2228,341:1420]

test_images = [
    "../ques1Images/1.png",
    "../ques1Images/5.png",
    "../ques1Images/3.png",
    "../ques1Images/2.png",
    "../ques1Images/10.png",
    "../ques1Images/9.png",
    "../ques1Images/6.png",
    "../ques1Images/8.png",
    "../ques1Images/4.png",
    "../ques1Images/7.png"
]

correlations = []

for img in test_images:
    testImg = cv2.imread(img)
    croppedTestImg = testImg[714:2228,341:1420]
    plt.imshow(croppedTestImg)
    plt.show()
    X = croppedTestImg - cropped_img
    ssd = sum(X[:]**2)
    correlations.append(ssd)


#[0,0,0] correlation means perfect match
print(correlations)


cv2.imshow("image of interest",interest_img)
plt.imshow(cropped_img)
plt.show()