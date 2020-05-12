import cv2
import numpy as np
import argparse
import time
import threading
from object_detection import load_yolo, image_detect, detect_objects, draw_labels, detect_objects_in_frame
from translate_and_say import ts


finished = False
frame = None
result = None

def process_image():
    global finished
    global frame
    global result
    global model
    global classes
    global colors
    global output_layers
    while True:
        if finished:
            break
        if frame is not None:
            #print("Process")
            #try:
                #result, texts = detect(frame, "frozen_east_text_detection.pb")
            result, texts = detect_objects_in_frame(frame, model, classes, colors, output_layers)
            (unique, counts) = np.unique(texts, return_counts=True)
            texts2 = []
            for i in range(len(unique)):
                if counts[i] > 1:
                    texts2.append(str(counts[i]) + " " + unique[i] + "s")
                else:
                    texts2.append(unique[i])
            if len(texts2) > 0:
                to_say = ", ".join(texts2)
                print(to_say)
                ts(to_say)
            #except:
            #    print("ERROR")
            frame = None

model, classes, colors, output_layers = load_yolo("yolo/yolov3.weights", "yolo/yolov3.cfg", "yolo/coco.names")


thread = threading.Thread(target=process_image, args=())
thread.start()

input_file = "video/terminator.mp4"
video = cv2.VideoCapture(0)

while True:
    success, image = video.read()
    if not success:
        finished = True
        break
    h, w = image.shape[:2]
    w1 = 800
    scale = w1 / w
    h1 = int(h * scale)
    new_size = (w1, h1)
    image = cv2.resize(image, new_size)
    if frame is None:
        frame = image
    cv2.imshow("Image", image)
    if result is not None:
        cv2.imshow("Result", result)
    if cv2.waitKey(50) == ord('q'):
        finished = True
        break

