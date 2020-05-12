import cv2
import numpy as np
import random

image = np.zeros((600, 600, 3), dtype=np.uint8)

def serp(image, p1, p2, n):
    w = (p2[0] - p1[0]) // 3
    h = (p2[1] - p1[1]) // 3
    k = random.randint(0, 8)
    for i in range(3):
        for j in range(3):
            pp1 = (p1[0] + w * i, p1[1] + h * j)
            pp2 = (p1[0] + w * (i + 1), p1[1] + h * (j + 1))
            if i * 3 + j == k:
                cv2.rectangle(image, pp1, pp2, (255, 255, 255), -1)
            elif n > 0:
                serp(image, pp1, pp2, n-1)

def dragon(image, p1, p2, n):
    if n == 0:
        cv2.line(image, p1, p2, (255, 255, 255))
    else:
        p0 = ((p2[0] + p1[0]) / 2, (p2[1] + p1[1]) / 2)
        a = (p0[0] - p1[0], p0[1] - p1[1])
        sign = 1 if random.randint(0, 1) == 0 else -1
        norm = (a[1] * sign, -a[0] * sign)
        p3 = (int(p0[0] + norm[0]), int(p0[1] + norm[1]))
        dragon(image, p1, p3, n-1)
        dragon(image, p3, p2, n-1)


#dragon(image, (150, 300), (450, 300), 15)

serp(image, (000, 000), (600, 600), 4)
cv2.imshow("IMAGE", image)
cv2.waitKey()