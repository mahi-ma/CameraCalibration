import cv2 

face_cascade = cv2.CascadeClassifier('./haarcascade_frontal_default.xml')

bcard = cv2.imread("./business_card.jpeg")

gray = cv2.cvtColor(bcard, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
for (x,y,w,h) in faces:
    # To draw a rectangle in a face 
    cv2.rectangle(bcard,(x,y),(x+w,y+h),(255,255,0),2) 
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = bcard[y:y+h, x:x+w]

# Display an image in a window
cv2.imshow('img',bcard)
cv2.waitKey(0)
cv2.destroyAllWindows()