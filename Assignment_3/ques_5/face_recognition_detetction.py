#train a model for face classification
import tensorflow;

#face detection
import os
import time
import cv2
import imutils

from tensorflow.keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator()

gen = idg.flow_from_directory("face_images",batch_size=100,target_size=(200,200))

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Dropout

resmodel = ResNet50(include_top=False, input_shape=(200, 200, 3))

resmodel.trainable = False

flat = Flatten()(resmodel.output)
d1 = Dense(1000, activation="relu")(flat)
d2 = Dense(500, activation="relu")(d1)
d3 = Dense(200, activation="relu")(d2)
drop1 = Dropout(.2)(d3)
d5 = Dense(100, activation="relu")(d3)
drop2 = Dropout(.2)(d5)
out_layer = Dense(3, activation="softmax")(drop2)

model = Model(resmodel.input, out_layer)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(gen, steps_per_epoch=18, epochs=25)

detectorPaths = {
    
    "face": "haarcascade_frontal_default.xml",
    # "smile": "smile.xml",
}

print("[INFO] loading haar cascades...")
detectors = dict()
 
for (name, path) in detectorPaths.items():
    detectors[name] = cv2.CascadeClassifier(path)

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

def getClass(index):
  keyDict = gen.class_indices
  reversedDict = dict()
  for key in keyDict:
      val = keyDict[key]
      reversedDict[val] = key
  return reversedDict[index]

while True:
    _,frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = detectors["face"].detectMultiScale(
        gray, 1.3,5)

    for (fX, fY, fW, fH) in faceRects:
        faceROI = gray[fY:fY + fH, fX:fX + fW]
        predictedPerson = model.predict(np.array([cv2.resize(faceROI,(200,200))]))
        result=getClass(predictedPerson.argmax(axis=1)[0])
        cv2.rectangle(frame, (fX, fY-20), (fX + 20, fY + 20),
                              (0, 255, 0), 2)
        cv2.putText(frame, result, (fX, fY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # smileRects = detectors["smile"].detectMultiScale(
        #     faceROI, scaleFactor=1.1, minNeighbors=10,
        #     minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
        # for (sX, sY, sW, sH) in smileRects:
        #     ptA = (fX + sX, fY + sY)
        #     ptB = (fX + sX + sW, fY + sY + sH)
        #     cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                      (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()