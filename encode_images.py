import face_recognition
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pickle
import cv2

#Set dataset path (where images are stored)
dataset_path = r"E:\face"  # Change this if needed
save_folder = r"E:\trained_face"  # Change this to your custom folder

# Create the folder if it does not exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Lists to store trained encodings and names
known_face_encodings = []
known_face_names = []

print("🔹 Training data from images...")

#Loop through each folder (each person's images)
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue  # Skip if not a folder

    # Process each image in the folder
    for filename in os.listdir(person_folder):
        img_path = os.path.join(person_folder, filename)

        try:
            # Load image and get face encodings
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            # If at least one face is found, save encoding
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"Training completed! {len(known_face_names)} faces were recognized.")

#Save trained data to custom folder
save_path = os.path.join(save_folder, "face_encodings.pickle")
#save_path = os.path.join(save_folder, "face_encodings.pickle")

with open(save_path, "wb") as file:
    pickle.dump((known_face_encodings, known_face_names), file)

print(f"Data saved to {save_path}")
