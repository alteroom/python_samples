import math
import numpy as np
import cv2

import pytesseract
from PIL import Image

def detect_fast(original):
    kernel = np.ones((3,3), dtype=np.uint8)

    orig = original.copy()
    image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)


    #image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 15, 5)
    cv2.imshow("binary", image)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(image, cv2.MORPH_DILATE, close_kernel, iterations=1)
    cv2.imshow("dilated", dilated)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grad = cv2.morphologyEx(dilated, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("grad 2", grad)

    #grad = cv2.morphologyEx(grad, cv2.MORPH_OPEN, close_kernel, iterations=10)
    #cv2.imshow("opened 4", grad)
    result_images = []
    text_images = []

    #result_images.append(image)
    #result_images.append(grad)
    
    contours, _ = cv2.findContours(grad, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    height, width = grad.shape[:2]
    digit_size = width / 10
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / h
        max_size = max(w, h)
        min_size = min(w, h)
        if ratio < 0.25 or ratio > 4 or w > digit_size or h > digit_size or max_size < digit_size / 2 or min_size < digit_size / 4:
            continue
        img = image[y:y+h, x:x+w]
        border = 20
        img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
        text_images.append(img)
        #cv2.imshow("img", img)
        #cv2.waitKey(500)
        cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 1)
    
    cv2.imshow("digits", orig)
    
    texts = []
    tess_config = "-l eng --psm 10"
    #tess_config = "-l eng --psm 10 -c tessedit_char_whitelist=0123456789"
    
    for i, img in enumerate(text_images):
        #img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
        result_images.append(img)
        filename = f"images/text_{i}.png"
        cv2.imwrite(filename, img)
        
        text = pytesseract.image_to_string(Image.open(filename), config=tess_config)
        if len(text) > 2:
            texts.append(text.lower())
        print(i, text)
    """
    return result_images, texts
    """

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.hypot(x2-x1, y2-y1)

filename = "images/sudoku1.jpg"
image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)

#image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LANCZOS4)
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

canny1 = cv2.Canny(gray, 50, 150)
#cv2.imshow("gray", gray)
cv2.imshow("canny1", canny1)

contours, _ = cv2.findContours(canny1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_id = -1
max_area = 0
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > max_area and area > 0:
        max_area = area
        max_id = i

if max_id >= 0:
    cv2.drawContours(image, contours, max_id, (0, 255, 0), thickness=1)

    cnt = contours[max_id]
    rect = cv2.minAreaRect(cnt)
    center, _, _ = rect

    LT = (center[0], center[1])
    RT = (center[0], center[1])
    RB = (center[0], center[1])
    LB = (center[0], center[1])
    cx, cy = center
    for point in cnt:
        x, y = point[0]
        d = distance((x, y), center)
        # LT
        if x <= cx and y <= cy and d > distance(LT, center):
            LT = (x, y)
        # RT
        if x > cx and y <= cy and d > distance(RT, center):
            RT = (x, y)
        # RB
        if x > cx and y > cy and d > distance(RB, center):
            RB = (x, y)
        # LB
        if x <= cx and y > cy and d > distance(LB, center):
            LB = (x, y)

    cv2.circle(image, (int(cx), int(cy)), 3, (0, 255, 0), thickness=2)

    cv2.circle(image, LT, 3, (0, 255, 255), thickness=2)
    cv2.circle(image, RT, 3, (0, 255, 255), thickness=2)
    cv2.circle(image, RB, 3, (0, 255, 255), thickness=2)
    cv2.circle(image, LB, 3, (0, 255, 255), thickness=2)
    maxWidth = 400
    maxHeight = 400
    src = np.array(
        [
            [LT[0], LT[1]],
            [RT[0], RT[1]],
            [RB[0], RB[1]],
            [LB[0], LB[1]]
        ], dtype = "float32")
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype = "float32")
    # src (54, 63) (369, 51) (391, 391) (25, 387)
    # dst (0, 0)   (399, 0)  (399, 399) (0, 399)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))
    cv2.imshow("warped", warped)

    #result, texts = detect_fast(warped)
    detect_fast(warped)


cv2.imshow("image", image)

cv2.waitKey()