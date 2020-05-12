import cv2
import numpy as np
from math import sqrt

def preprocess(frame, bg, thresh):
    diff = cv2.absdiff(frame, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    gray[gray < thresh] = 0
    gray[gray >= thresh] = 255
    
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, small_kernel, iterations=1)

    big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, big_kernel, iterations=1)

    return gray

def distance(b1, b2):
    _, _, _, h1, cx1, cy1 = b1
    _, _, _, h2, cx2, cy2 = b2
    h = min(h1, h2)
    d = sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
    return d / h

input_file = "video/vtest.mp4"
video = cv2.VideoCapture(input_file)
bg = cv2.imread("images/bg.jpeg")

thresh = 40

max_distance = 0.7

while True:
    success, frame = video.read()
    if not success:
        break
    gray = preprocess(frame, bg, thresh)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    
    for cnt in contours:
        box = cv2.boundingRect(cnt)
        x, y, w, h = box
        if h < 30:
            continue
        cx = x + w // 2
        cy = y + h // 2
        boxes.append((x, y, w, h, cx, cy))

    bad = set()
    links = []
    for i in range(len(boxes)-1):
        for j in range(i+1, len(boxes)):
            b1 = boxes[i]
            b2 = boxes[j]
            d = distance(b1, b2)
            if d < max_distance:
                bad.add(i)
                bad.add(j)
                links.append((i, j))

    for i, box in enumerate(boxes):
        x, y, w, h, cx, cy = box
        if w / h > 0.5:
            bad.add(i)
        if i in bad:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 128, 255), thickness=-1)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for link in links:
        i, j = link
        _, _, _, _, cx1, cy1 = boxes[i]
        _, _, _, _, cx2, cy2 = boxes[j]
        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)

    cv2.imshow("Video", frame)
    cv2.imshow("MASK", gray)
    
    if cv2.waitKey(1) == ord('q'):
        break




