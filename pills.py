import cv2
import numpy as np

image = cv2.imread("images/pills/5.jpg")
h, w, c = image.shape
w1 = 400
scale = w1 / w
h1 = int(h * scale)
new_size = (w1, h1)
image = cv2.resize(image, new_size)

pills = cv2.inRange(image, (35, 50, 90), (180, 210, 250))
pills = cv2.bitwise_not(pills)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
pills = cv2.morphologyEx(pills, cv2.MORPH_OPEN, kernel, iterations=3)
pills = cv2.morphologyEx(pills, cv2.MORPH_CLOSE, kernel, iterations=1)

cv2.imshow("1", pills)

contours, _ = cv2.findContours(pills, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    center, size, angle = cv2.minAreaRect(c)
    center = (int(center[0]), int(center[1]))
    radius = size[0] // 4 + size[1] // 4
    ratio = size[0] / size[1]
    min_size = min(size[0], size[1])
    if min_size < 20:
        continue
    if (ratio < 0.75) or (ratio > 1.3):
        continue
    cv2.circle(image, center, int(radius), (0, 255, 0), 3)
    print(ratio)
cv2.imshow("Pills", pills)
cv2.imshow("Image", image)
cv2.waitKey()