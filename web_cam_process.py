import threading
import time
import cv2
import numpy as np

finished = False
frame = None
result = ""
def process():
    global frame
    global result
    global finished
    count = 0
    while True:
        if finished:
            break
        if frame is not None:
            print("start process")
            time.sleep(5)
            count += 1
            result = f"Processed {count}"
            print("end process")
            frame = None


thread = threading.Thread(target=process, args=())
thread.start()

video = cv2.VideoCapture(0)
while True:
    success, image = video.read()
    if not success:
        break
    h, w = image.shape[:2]
    w1 = 500
    scale = w1 / w
    h1 = int(h * scale)
    new_size = (w1, h1)
    image = cv2.resize(image, new_size)
    if frame is None:
        frame = image.copy()
    
    cv2.putText(image, result, (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    
    if cv2.waitKey(1) == ord('q'):
        finished = True
        break