import face_recognition
import cv2 as cv
import pickle
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

with open("E:/trained_face/face_encodings.pickle", "rb") as file:
    known_face_encodings, known_face_names = pickle.load(file)
print(f"Loaded {len(known_face_names)} trained faces!")

json_file = "people.json"

try:
    with open(json_file, "r", encoding="utf-8") as file:
        person_info = json.load(file)
    print(f"Loaded person information from {json_file}")
except FileNotFoundError:
    print(f"Error: {json_file} not found")
    person_info = {}

video = cv.VideoCapture(0)

while True:
    ret, frame = video.read()

    if not ret:
        print("Could not access webcam")
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (x, y, w, h), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        detail = ""
        
        face_instance =face_recognition.face_distance(known_face_encodings, face_encoding)
        best_index = np.argmin(face_instance)

        if matches[best_index]:
            name = known_face_names[best_index]
            if name in person_info:
                details = f"Age: {person_info[name]['age']} | Job: {person_info[name]['job']} | Location: {person_info[name]['location']} | Email: {person_info[name]['E-mail']}"


        cv.rectangle(frame, (h, x), (y, w), (0, 255, 0), 2)

        cv.putText(frame, name, (h, w + 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if details:
              cv.putText(frame, details, (h, w + 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv.imshow('Face recognition', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
