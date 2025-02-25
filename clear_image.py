import sys
import cv2 as cv
import numpy as np
import face_recognition
import json
import pickle
import os
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLineEdit
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class FaceRecognition(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_data()  # Load face encodings and person data
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.cap = cv.VideoCapture(0)

    def initUI(self):
        self.setWindowTitle("Face Recognition")
        self.setGeometry(100, 100, 900, 600)

        # Camera display
        self.camera = QLabel(self)
        self.camera.setFixedSize(500, 640)

        # Create input fields for Name, Age, Job, and Email
        self.name_table = self.create_table()
        self.age_table = self.create_table()
        self.job_table = self.create_table()
        self.email_table = self.create_table()

        # Stop button
        self.stop_button = QPushButton("STOP", self)
        self.stop_button.clicked.connect(self.close_app)

        # Layout: Display tables next to the camera feed
        layout = QHBoxLayout(self)
        layout.addWidget(self.camera)

        # Create a vertical layout to stack tables
        table_layout = QVBoxLayout()
        table_layout.addLayout(self.create_row("Name", self.name_table))
        table_layout.addLayout(self.create_row("Age", self.age_table))
        table_layout.addLayout(self.create_row("Job", self.job_table))
        table_layout.addLayout(self.create_row("Email", self.email_table))
        table_layout.addWidget(self.stop_button)

        layout.addLayout(table_layout)
        self.setLayout(layout)

    def create_table(self):
        """Creates a read-only QLineEdit input field for displaying data."""
        input_field = QLineEdit(self)
        input_field.setReadOnly(True)
        input_field.setPlaceholderText("Unknown")
        return input_field

    def create_row(self, label_text, input_field):
        """Creates a row layout with a label and input field."""
        row_layout = QHBoxLayout()
        label = QLabel(label_text, self)
        label.setFixedWidth(50)  # Ensures the label is visible
        row_layout.addWidget(label)
        row_layout.addWidget(input_field)
        return row_layout

    def load_data(self):
        """Loads trained face encodings and person information from files."""
        try:
            with open("E:/trained_face/face_encodings.pickle", "rb") as file:
                self.known_face_encodings, self.known_face_names = pickle.load(file)
            print(f"Loaded {len(self.known_face_names)} trained faces.")
        except FileNotFoundError:
            print("Error: Face encodings file not found.")
            self.known_face_encodings, self.known_face_names = [], []

        json_file = "people.json"
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                self.person_info = json.load(file)
            print(f"Loaded person information from {json_file}.")
        except FileNotFoundError:
            print("Error: person.json not found.")
            self.person_info = {}

    def update_frame(self):
        """Captures frames, detects faces, and updates the UI with recognized information."""
        ret, frame = self.cap.read()
        if not ret:
            print("Could not access webcam")
            self.timer.stop()
            return

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #rgb_frame = cv.cvtColor(gray, cv.COLOR_GBR2GRAY)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        name, age, job, email = "Unknown", "Unknown", "Unknown", "Unknown"

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                if name in self.person_info:
                    person = self.person_info[name]
                    age = person.get('age', 'Unknown')
                    job = person.get('job', 'Unknown')
                    email = person.get('E-mail', 'Unknown')

            # Draw bounding box around the face
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display name below the face
            cv.putText(frame, name, (left, bottom + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Update UI fields
        self.name_table.setText(name)
        self.age_table.setText(str(age))
        self.job_table.setText(job)
        self.email_table.setText(email)

        # Convert frame to QImage for display
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera.setPixmap(QPixmap.fromImage(q_img))

    def close_app(self):
        """Stops the camera and closes the application."""
        self.timer.stop()
        self.cap.release()
        cv.destroyAllWindows()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognition()
    window.show()
    sys.exit(app.exec())
