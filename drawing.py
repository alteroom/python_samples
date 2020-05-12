import cv2
import numpy as np

image = np.zeros((400, 600, 3), dtype=np.uint8)


cv2.rectangle(image, (200, 100), (400, 300), (255, 0, 0), 1, lineType=cv2.LINE_AA)
cv2.line(image, (200, 100), (400, 300), (255, 0, 255), 4, lineType=cv2.LINE_8)
cv2.circle(image, (300, 300), 95, (0, 255, 0), 10)

cv2.putText(image, "DRAWING FUNCTIONS", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
cv2.putText(image, "DRAWING FUNCTIONS", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
cv2.putText(image, "DRAWING FUNCTIONS", (20, 60), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255))


cv2.imshow("IMAGE", image)
cv2.waitKey()