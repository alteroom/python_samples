import cv2
import time
# Load the cascade
face_cascade = cv2.CascadeClassifier('haar.xml')

video = cv2.VideoCapture(0)

while True:
    # Read the input image
    success, img = video.read()
    if not success:
        break
    h, w = img.shape[:2]
    w1 = 800
    scale = w1 / w
    h1 = int(h * scale)
    new_size = (w1, h1)
    img = cv2.resize(img, new_size)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    start_time = time.time()
    faces = face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(80, 80))
    duration = time.time() - start_time
    fps = int(1 / duration)
    cv2.putText(img, f"FPS: {fps}", (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break