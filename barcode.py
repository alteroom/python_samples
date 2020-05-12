import cv2
import numpy as np


kernel = np.ones((3,3), dtype=np.uint8)

original = cv2.imread("images/barcode2.jpg")

image = cv2.imread("images/barcode2.jpg", cv2.IMREAD_GRAYSCALE)
image[image >= 120] = 255
image[image < 120] = 0

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
#cv2.imshow("IMAGE 1", image)
cv2.imshow("grad 2", grad)

close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, close_kernel)
cv2.imshow("closed 3", grad)
grad = cv2.morphologyEx(grad, cv2.MORPH_OPEN, close_kernel, iterations=10)
cv2.imshow("opened 4", grad)

contours, _ = cv2.findContours(grad, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(original, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 3)

cv2.imshow("Detected", original)


cv2.waitKey()