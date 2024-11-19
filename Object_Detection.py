import cv2
import numpy as np

cfgModel = "B:/object fix 2/yolov3.1.cfg";
weightsModel = "B:/object fix 2/yolov3.1.weights";

net = cv2.dnn.readNetFromDarknet(cfgModel, weightsModel);

classes = []
with open("B:/object fix 2/coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    # 320 x 320 (high speed, less accuracy),
    # 416 x 416 (moderate speed, moderate accuracy)
    # 608 x 608 (less speed, high accuracy)

    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # Get the confidence percentage
            confidence_percentage = round(confidences[i] * 100, 2)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + str(confidence_percentage) + "%", (x, y + 20), font, 2, (255, 255, 255), 2)

    cv2.imshow('FPVcamera', img)
    key = cv2.waitKey(5)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()