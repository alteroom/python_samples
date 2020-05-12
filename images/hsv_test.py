import numpy as np
import cv2

filename = "images/elon.jpg"
image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
image = cv2.resize(image, None, fx=0.5, fy=0.5)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("image", image)
h, s, v = cv2.split(hsv)
cv2.imshow("h", h)
cv2.imshow("s", s)
cv2.imshow("v", v)

v = cv2.multiply(v, None, 2)
#v[:,:] = v[:,:] * 2

hsv2 = cv2.merge([h, s, v])
image2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
cv2.imshow("image2", image2)

# h 0..180
# s 0..255
# v 0..255

#BGR 0, 0, 0 black
#s=0
#v=0
# white
#s =0
# v=255 
# red 255, 0, 0
#h=0
#s=1
#v=1

cv2.waitKey()
