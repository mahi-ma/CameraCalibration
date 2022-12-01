import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np


#this function for creating rectangle with different colours
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = color[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_output_layers(net):
    layer = net.getLayerNames()
    output_layers = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# def apply_filter(img, tleft, w_width, w_height):
#     x = tleft[0]
#     y = tleft[1]
#     return cv2.GaussianBlur(img[y:y+window_size, x:x+window_size], (51,51), 0)



# command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,help = 'path to input image')
ap.add_argument('-c', '--config', required=True,help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,help = 'path to text file containing class names')
args = ap.parse_args()

img = cv2.imread("sample1.png")
height = img.shape[0]
width = img.shape[1]

scale = 0.00392

# read class names from text file
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# # generate different colors for different classes 
color = np.random.uniform(0, 255, size=(len(classes), 3))
blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
pre_def = cv2.dnn.readNet(args.weights, args.config)

pre_def.setInput(blob) #set input blob for the network
outs = pre_def.forward(get_output_layers(pre_def))

conf_threshold = 0.5
nms_threshold = 0.4
class_ids = []
confidences = []
boxes = []

for out in outs:  #detecting from each layer
    for det in out:
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            x_cen = int(det[0] * width)
            y_cen = int(det[1] * height)
            w = int(det[2] * width)
            h = int(det[3] * height)
            x = x_cen - w / 2
            y = y_cen - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold) #non max method

for i in indices: #going through indices
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

   
cv2.imshow("object detection", img)

# wait until any key is pressed
cv2.waitKey()
#  # save output image to disk
# cv2.imwrite("object-detection.jpg", img)

cv2.destroyAllWindows()





#sliding window method

# img_copy = img.copy()

# cord = [0,0]
# window_size = 100
# dir = "right"

# while True:
#     if dir == "right":
#         cord[0] += 1
#     elif dir == "left":
#         cord[0] -= 1
    
#     bottom_cord = (cord[0] + window_size, cord[1]+window_size)
#     output_win = apply_filter(img, cord, window_size, window_size)

#     x = cord[0]
#     y = cord[1]
#     img_copy[y:y+window_size, x:x+window_size] = output_win

#     cv2.rectangle(img_copy, (cord[0], cord[1]), bottom_cord, (0,255,0), 2)

#     cv2.imshow("Img", img_copy)
#     cv2.waitKey(5)

#     img_copy = img.copy()

#     if bottom_cord[0] >= width:
#         dir = "left"
#         cord[1] += 30
#     elif bottom_cord[0] <= 0:
#         dir = "right"
#         cord[1] += 30
    
#     if bottom_cord[0] >= width and bottom_cord[1] >= height:
#         break


# cv2.destroyAllWindows()
    


