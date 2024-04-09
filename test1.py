import cv2
import numpy as np
import tensorflow as tf

# Load the saved gender classification model
model_filename = "gender_classification_model.h5"  # Replace with the path to your saved gender model
gender_model = tf.keras.models.load_model(model_filename)

# Load the face detection model from OpenCV (you can change the path accordingly)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to preprocess an image
def preprocess_image(image, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    image = cv2.resize(image, image_size)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to predict gender from an image
def predict_gender(image, model):
    image_size = (48, 48)  # Adjust the image size based on your model's input shape
    preprocessed_image = preprocess_image(image, image_size)
    gender_labels = ["Male", "Female"]
    predictions = model.predict(preprocessed_image)
    predicted_gender = gender_labels[np.argmax(predictions)]
    return predicted_gender

# Open a connection to the webcam (you may need to change the index)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_roi = frame[y:y+h, x:x+w]

        # Predict gender for the face region
        predicted_gender = predict_gender(face_roi, gender_model)

        # Display the predicted gender on the frame
        cv2.putText(frame, f"Gender: {predicted_gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame with the predicted genders and face rectangles
    cv2.imshow('Gender Classification', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
