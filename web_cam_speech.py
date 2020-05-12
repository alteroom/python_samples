import cv2
import numpy as np
import threading
import time
from detect_text import detect
from text_detector2 import detect_fast
from translate_and_say import ts

finished = False
frame = None
result = None

def process_image():
    global finished
    global frame
    global result
    while True:
        if finished:
            break
        if frame is not None:
            #print("Process")
            #try:
                #result, texts = detect(frame, "frozen_east_text_detection.pb")
            result, texts = detect_fast(frame)
            if len(texts) > 0:
                to_say = " ".join(texts)
                ts(to_say)
            #except:
            #    print("ERROR")
            frame = None

thread = threading.Thread(target=process_image, args=())
thread.start()

input_file = "video/alphabet.mp4"
video = cv2.VideoCapture(0)#input_file)

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
        for i, img in enumerate(result):
            cv2.imshow(f"Result {i}", img)
    if cv2.waitKey(50) == ord('q'):
        finished = True
        break