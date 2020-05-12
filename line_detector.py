import numpy as np
import cv2

filename = "images/sudoku1.jpg"
image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)

#image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LANCZOS4)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

canny1 = cv2.Canny(gray, 50, 150)
#cv2.imshow("gray", gray)
cv2.imshow("canny1", canny1)
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 100  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments
lines = cv2.HoughLinesP(
    canny1, rho, theta, threshold,
    minLineLength=min_line_length,
    maxLineGap=max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("lines", image)

cv2.waitKey()