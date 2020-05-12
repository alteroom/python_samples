import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v)) 
    upper = int(min(255, (1.0 + sigma) * v)) 
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

filename = "images/pills/14.jpg"
#filename = "images/plates/PLACARD-PT017A9-1584459451-0.jpg"
#filename = "images/plates/EMPTY PLACARDS-PT156F7-1584460198-0.jpg"
image = cv2.imread(filename)
h, w, c = image.shape
w1 = 400
scale = w1 / w
h1 = int(h * scale)
new_size = (w1, h1)
image = cv2.resize(image, new_size)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#canny = auto_canny(blurred, 0.6)
canny = cv2.Canny(image, 50, 200)

contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    center, size, angle = cv2.minAreaRect(c)
    center = (int(center[0]), int(center[1]))
    radius = size[0] // 4 + size[1] // 4
    min_size = min(size[0], size[1])
    if min_size < 20:
        continue
    ratio = size[0] / size[1]
    if (ratio < 0.3) or (ratio > 2.5):
        continue
    cv2.circle(image, center, int(radius), (0, 255, 0), 3)
    print(ratio)


cv2.imshow("canny", canny)
#cv2.imshow("blurred", blurred)
cv2.imshow("Image", image)

cv2.waitKey()