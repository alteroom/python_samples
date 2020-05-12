import cv2
import numpy as np
import time

def generate_mask(sz):
    mask = np.zeros((sz, sz), dtype=np.uint8)
    k = 0.6
    left = int(sz * k) // 2
    right = sz // 2
    cv2.circle(mask, (sz // 2, sz // 2), left, 255, -1)
    for r in range(left, right):
        color = 255 * (sz // 2 - r) / int(sz * (1 - k) / 2)
        cv2.circle(mask, (sz // 2, sz // 2), r, color, 2)
    return mask

def swap_faces(image, box1, box2, flip=False):
    h, w, c = image.shape
    mask = generate_mask(100)
    #cv2.imshow("mask", mask)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    dy = int(h1 * 0.1)
    y1 = max(y1 - dy, 0)
    h1 = min(h1 + 2*dy, h)
    dy = int(h2 * 0.1)
    y2 = max(y2 - dy, 0)
    h2 = min(h2 + 2*dy, h)

    face1 = image[y1:y1+h1, x1:x1+w1]
    face2 = image[y2:y2+h2, x2:x2+w2]
    if flip:
        face1 = cv2.flip(face1, 1)
        face2 = cv2.flip(face2, 1)

    swap1 = cv2.resize(face2, (w1, h1))
    swap2 = cv2.resize(face1, (w2, h2))
    
    mask1 = cv2.resize(mask, (w1, h1))
    mask2 = cv2.resize(mask, (w2, h2))
    #cv2.imshow("mask1", swap1)
    #cv2.imshow("mask2", swap2)
    
    for c in range(0, 3):
        image[y1:y1+h1, x1:x1+w1, c] = swap1[:h1, :w1, c] * (mask1 / 255) + image[y1:y1+h1, x1:x1+w1, c] * (1 - (mask1 / 255))
        image[y2:y2+h2, x2:x2+w2, c] = swap2[:h2, :w2, c] * (mask2 / 255) + image[y2:y2+h2, x2:x2+w2, c] * (1 - (mask2 / 255))


# Load the cascade
face_cascade = cv2.CascadeClassifier('haar.xml')
# Read the input image
inputfile = 'images/swap2.jpg'
name = inputfile.split(".")[0]
ext = inputfile.split(".")[1]
outfile = name + "_swapped." + ext
img = cv2.imread(inputfile)
start_time = time.time()
h, w = img.shape[:2]
w1 = 800
scale = w1 / w
h1 = int(h * scale)
new_size = (w1, h1)
img = cv2.resize(img, new_size)

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces

faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
faces = sorted(faces, key=lambda x: x[3], reverse=True)

if len(faces) >= 2:
    face0 = faces[0]
    face1 = faces[1]
    swap_faces(img, face0, face1, flip=True)
    #cv2.imwrite(outfile, img)
# Draw rectangle around the faces
#for (x, y, w, h) in faces:
#    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
duration = time.time() - start_time
print(duration)

cv2.imshow('img', img)

cv2.waitKey()

