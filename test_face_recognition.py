import face_recognition
import cv2
import time

image = face_recognition.load_image_file("images/arnold.jpg")

image = cv2.resize(image, None, fx=0.5, fy=0.5)

start = time.time()
face_locations = face_recognition.face_locations(image)
end = time.time() - start
print(f"Face Locations {end} s")

bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for face in face_locations:
    x, y, x2, y2 = face
    cv2.rectangle(bgr, (x, y), (x2, y2), (0, 255, 0))


cv2.imshow("image", bgr)
cv2.waitKey(1)
start = time.time()
face_landmark = face_recognition.face_landmarks(image, face_locations)
end = time.time() - start
print(f"Face Landmarks {end} s")
for face in face_landmark:
    for key, points in face.items():
        print(key, len(points))
        for i in range(len(points)-1):
            cv2.line(bgr, points[i], points[i+1], (0, 0, 255))

cv2.imshow("image", bgr)
cv2.waitKey()
