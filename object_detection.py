import cv2
import numpy as np
import argparse
import time

def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

#YOLO You only look once
def load_yolo(weights_file, cfg_file, coco_names_file):
    net = cv2.dnn.readNet(weights_file, cfg_file)
    classes = []
    with open(coco_names_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=0.00392,
        size=(320, 320),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            #print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    labels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            labels.append(label)
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    #cv2.imshow("Image", img)
    return img, labels

def image_detect(img_path, model, classes, colors, output_layers): 
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    img, _ = draw_labels(boxes, confs, colors, class_ids, classes, image)
    return img

def detect_objects_in_frame(frame, model, classes, colors, output_layers):
    image = frame.copy()
    height, width, channels = image.shape
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    img, labels = draw_labels(boxes, confs, colors, class_ids, classes, image)
    return img, labels


if __name__ == "__main__":
    model, classes, colors, output_layers = load_yolo("yolo/yolov3.weights", "yolo/yolov3.cfg", "yolo/coco.names")
    img = image_detect("images/elon.jpg", model, classes, colors, output_layers)
    cv2.imshow("elon", img)
    img = image_detect("images/swap1.jpg", model, classes, colors, output_layers)
    cv2.imshow("Tesla", img)
    img = image_detect("images/plates/MULTIPLE PLATES.jpg", model, classes, colors, output_layers)
    cv2.imshow("cars", img)
    cv2.waitKey()


