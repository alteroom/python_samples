import face_recognition
import cv2
import time

petrov_image = face_recognition.load_image_file("images/petrov.jpg")
petrov_face_encoding = face_recognition.face_encodings(petrov_image)[0]

ts_image = face_recognition.load_image_file("images/ts.jpg")
ts_face_encoding = face_recognition.face_encodings(ts_image)[0]

known_face_encodings = [
    petrov_face_encoding,
    ts_face_encoding
]
known_face_names = [
    "Petrov",
    "TS"
]

video = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    success, frame = video.read()
    if not success:
        break
    scale = 0.25
    unscale = 2
    image = cv2.resize(frame, None, fx=scale, fy=scale)
    rgb_small_frame = image[:, :, ::-1]

    #face_locations = face_recognition.face_locations(image)
    #face_landmark = face_recognition.face_landmarks(image, face_locations)

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            #face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #best_match_index = np.argmin(face_distances)
            #if matches[best_match_index]:
            #    name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #for face in face_landmark:
    #    for key, points in face.items():
    #        for i in range(len(points)-1):
    #            cv2.line(image, points[i], points[i+1], (0, 255, 0))

    #image = cv2.resize(image, None, fx=unscale, fy=unscale)


    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

