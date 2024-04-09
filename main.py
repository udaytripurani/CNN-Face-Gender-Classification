import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Define a function to load and preprocess images from a directory and split into train and validation
def load_and_split_images(directory, image_size, test_size=0.2, random_seed=42):
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

    x = np.array(images)
    y = np.array(labels)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_seed)

    return x_train, x_val, y_train, y_val

# Define the directory path for your dataset
dataset_directory = "C:/Users/Uday/Downloads/archive (8)/faces"  # Update with the path to your dataset folder

# Load and preprocess data, splitting it into training and validation sets
image_size = (48, 48)  # Adjust the image size based on your model's input shape
x_train, x_val, y_train, y_val = load_and_split_images(dataset_directory, image_size)

# Define the CNN model for gender classification
def create_gender_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Define the input shape and number of classes (genders)
input_shape = (48, 48, 1)  # Adjust the image size based on your model's input shape
num_classes = 2  # Number of gender classes (male and female)

# Create the model
gender_model = create_gender_model(input_shape, num_classes)

# Compile the model
gender_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 20
history = gender_model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val))

# Save the model
model_filename = "gender_model.h5"
gender_model.save(model_filename)
print(f"Model saved as {model_filename}")

# Evaluate the model on the test set
test_directory = "path/to/test/gender"  # Update with the path to your test data folder
x_test, y_test = load_and_split_images(test_directory, image_size)
test_loss, test_acc = gender_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Make predictions on the test set
y_pred = gender_model.predict(x_test)

# Convert one-hot encoded predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Generate a classification report and confusion matrix
print(classification_report(y_test, y_pred_labels))
print(confusion_matrix(y_test, y_pred_labels))
