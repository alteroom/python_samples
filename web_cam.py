import cv2
import numpy as np


#image = cv2.imread("images/tesla.jpg")

video = cv2.VideoCapture(0)
while True:
    success, image = video.read()
    if not success:
        break
    h, w = image.shape[:2]
    w1 = 500
    scale = w1 / w
    h1 = int(h * scale)
    new_size = (w1, h1)
    image = cv2.resize(image, new_size)
    blur = cv2.blur(image, (7, 7))
    cv2.imshow("Blur", blur)
    mblur = cv2.medianBlur(image, 7)
    cv2.imshow("Median blur", mblur)
    b, g, r = cv2.split(mblur)
    z = np.zeros_like(b)
    b[b < 127] = 0
    b[b >= 127] = 255
    g[g < 127] = 0
    g[g >= 127] = 255
    r[r < 127] = 0
    r[r >= 127] = 255
    new = cv2.merge((b, g, r))
    cv2.imshow("Image", image)
    cv2.imshow("New", new)
    if cv2.waitKey(1) == ord('q'):
        break
