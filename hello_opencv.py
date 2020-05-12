import cv2
import numpy as np

image = cv2.imread("images/tesla.jpg", cv2.IMREAD_GRAYSCALE)
h, w = image.shape[:2]
print(image.shape)
w1 = 100
scale = w1 / w
h1 = int(h * scale)
new_size = (w1, h1)
image = cv2.resize(image, new_size)
print(image.shape)
cv2.imshow("Original", image)
cv2.imwrite("images/tesla_preview.jpg", image)
cv2.waitKey()