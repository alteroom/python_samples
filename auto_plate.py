import cv2
import numpy as np

#filename = "images/plates/MULTIPLE PLATES.jpg"
#filename = "images/plates/PLACARD-PT017A9-1584459451-0.jpg"
filename = "images/plates/EMPTY PLACARDS-PT156F7-1584460198-0.jpg"
image = cv2.imread(filename)
h, w, c = image.shape
w1 = 800
scale = w1 / w
h1 = int(h * scale)
new_size = (w1, h1)
image = cv2.resize(image, new_size)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=1)
gradX = cv2.Sobel(black, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
cv2.imshow("gradX", gradX)

thresh = 30
gradX[gradX >= thresh] = 255
gradX[gradX < thresh] = 0
# close width
small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, small_kernel, iterations=1)
# open height
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
gradX = cv2.morphologyEx(gradX, cv2.MORPH_OPEN, open_kernel, iterations=1)
# open width
big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
gradX = cv2.morphologyEx(gradX, cv2.MORPH_OPEN, big_kernel, iterations=1)

contours, _ = cv2.findContours(gradX, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    ratio = w / h
    if ratio <= 3.5:
        continue
    if w < 40:
        continue
    print(ratio)
    cv2.rectangle(image, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 3)


cv2.imshow("Image", image)
cv2.imshow("Result", gradX)

cv2.waitKey()