import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Define a function to load and preprocess images from a directory
def load_and_preprocess_images(directory, image_size):
    images = []
    labels = []

    genders_dict = {
        "man": 0,
        "woman": 1,
    }

    for gender_folder in os.listdir(directory):
        gender_label = genders_dict[gender_folder]
        gender_path = os.path.join(directory, gender_folder)

        for image_file in os.listdir(gender_path):
            image_path = os.path.join(gender_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalize pixel values to the range [0, 1]
            images.append(image)
            labels.append(gender_label)

    return np.array(images), np.array(labels)

# Define the directory path for your test dataset
test_directory = "C:/Users/Uday/Downloads/archive (8)/faces/test"  # Update with the path to your test data folder

# Load and preprocess the test data
image_size = (48, 48)  # Adjust the image size based on your model's input shape
x_test, y_test = load_and_preprocess_images(test_directory, image_size)

# Normalize the test data (if not already normalized during loading)
x_test = x_test / 255.0

# Load the pre-trained gender classification model
model_filename = "gender_model.h5"  # Update with the path to your saved model
gender_model = load_model(model_filename)

# Evaluate the model on the test set
test_loss, test_acc = gender_model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

# Make predictions on the test set
y_pred = gender_model.predict(x_test)

# Convert one-hot encoded predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Generate a classification report and confusion matrix
print(classification_report(y_test, y_pred_labels))
confusion = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:")
print(confusion)
