import cv2
import numpy as np

image = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(image, (100, 50), (200, 250), 255, -1)
cv2.rectangle(image, (50, 100), (250, 200), 255, -1)

kernel = np.ones((3,3), dtype=np.uint8)
erode = cv2.erode(image, kernel, iterations=40)
dilation = cv2.dilate(image, kernel, iterations=40)

cv2.imshow("IMAGE", image)
#cv2.imshow("Erode", erode)
#cv2.imshow("dilation", dilation)

image = np.zeros((300, 300), dtype=np.uint8)
cv2.circle(image, (100, 100), 40, 255, -1)
cv2.rectangle(image, (200, 150), (280, 200), 255, -1)
#cv2.line(image, (100, 100), (200, 150), 255, 20)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=1)
cv2.imshow("IMAGE", image)
cv2.imshow("grad", grad)


cv2.waitKey()