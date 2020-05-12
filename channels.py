import cv2
import numpy as np

image = cv2.imread("images/tesla.jpg")
h, w = image.shape[:2]
print(image.shape)
w1 = 1000
scale = w1 / w
h1 = int(h * scale)
new_size = (w1, h1)
image = cv2.resize(image, new_size)
print(image.shape)
cv2.imshow("Original", image)

b, g, r = cv2.split(image)
z = np.zeros_like(b)

b[b < 127] = 0
b[b >= 127] = 255
g[g < 127] = 0
g[g >= 127] = 255
r[r < 127] = 0
r[r >= 127] = 255


cv2.imshow("Blue", cv2.merge((b, z, z)))
cv2.imshow("Green", cv2.merge((z, g, z)))
cv2.imshow("Red", cv2.merge((z, z, r)))

new = cv2.merge((b, g, r))
cv2.imshow("New", new)
cv2.waitKey()