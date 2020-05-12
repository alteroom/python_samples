import cv2
import time
# Load the cascade
face_cascade = cv2.CascadeClassifier('haar.xml')

video = cv2.VideoCapture(0)
old_faces = []
while True:
    # Read the input image
    success, img = video.read()
    if not success:
        break
    h, w = img.shape[:2]
    w1 = 1024
    scale = w1 / w
    h1 = int(h * scale)
    new_size = (w1, h1)
    img = cv2.resize(img, new_size)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    start_time = time.time()
    faces = face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(150, 150))
    duration = time.time() - start_time
    fps = int(1 / duration)
    cv2.putText(img, f"FPS: {fps}", (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    # Draw rectangle around the faces
    if len(faces) == 0:
        faces = old_faces
    else:
        old_faces = faces
    for (x, y, w, h) in faces:
        dx = int(w * 0.1)
        dy = int(h * 0.2)
        yy = max(y - dy, 0)
        xx = max(x - dx, 0)
        ww = w + 2*dx
        hh = h + 2*dy
        if yy + hh > h1:
            hh = h1 - yy
        if xx + ww > w1:
            ww = w1 - xx
        face = img[yy:yy+hh, xx:xx+ww]
        face1 = cv2.resize(face, (16, 16), interpolation=cv2.INTER_AREA)
        #face2 = cv2.resize(face, (16, 16), interpolation=cv2.INTER_AREA)
        face1 = cv2.resize(face1, (ww, hh), interpolation=cv2.INTER_NEAREST)
        #face2 = cv2.resize(face2, (ww, hh), interpolation=cv2.INTER_CUBIC)
        #face = cv2.blur(face, (51, 51))
        img[yy:yy+hh, xx:xx+ww] = face1
        img2 = img.copy()
        #img2[yy:yy+hh, xx:xx+ww] = face2
        #cv2.imshow('img2', img2)
        

    # Display the output
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break