import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('gender_classification_model.h5')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define image dimensions
image_height, image_width = 224, 224

# Create a function to preprocess the input frame
def preprocess_frame(frame):
    # Resize the frame to match the model's expected input size
    frame = cv2.resize(frame, (image_width, image_height))
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to the range [0, 1]
    frame = frame / 255.0
    # Expand dimensions to match the model's input shape
    frame = np.expand_dims(frame, axis=0)
    return frame

# Open a connection to the webcam (0 indicates the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y + h, x:x + w]

        # Preprocess the face
        preprocessed_face = preprocess_frame(face)

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(preprocessed_face)

        # Determine the gender label based on the prediction (0 for male, 1 for female)
        gender_label = "Male" if prediction < 0.5 else "Female"

        # Display the frame with the predicted gender label and rectangle around the detected face
        cv2.putText(frame, f'Gender: {gender_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with faces and gender labels
    cv2.imshow('Gender Classification', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
