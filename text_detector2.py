import cv2
import numpy as np
import pytesseract
from PIL import Image

def detect_fast(original):

    kernel = np.ones((3,3), dtype=np.uint8)

    image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 15, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    #v2.imshow("grad 2", grad)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, close_kernel)
    #cv2.imshow("closed 3", grad)
    #grad = cv2.morphologyEx(grad, cv2.MORPH_OPEN, close_kernel, iterations=10)
    #cv2.imshow("opened 4", grad)
    result_images = []
    text_images = []

    result_images.append(image)
    result_images.append(grad)
    
    contours, _ = cv2.findContours(grad, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / h
        if ratio < 10 or w < 100:
            continue
        img = image[y:y+h, x:x+w]
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
        text_images.append(img)
        #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 1)
    
    texts = []
    tess_config = "-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    for i, img in enumerate(text_images):
        #img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
        result_images.append(img)
        filename = f"images/text_{i}.png"
        cv2.imwrite(filename, img)
        
        text = pytesseract.image_to_string(Image.open(filename), config=tess_config)
        if len(text) > 2:
            texts.append(text.lower())
        print(i, text)
    
    return result_images, texts