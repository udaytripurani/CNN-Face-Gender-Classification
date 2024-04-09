import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your preprocessed dataset
# Replace 'data_directory' with the path to your dataset directory
data_directory = 'C:/Users/Uday/Downloads/archive (8)/faces'

# Define image dimensions and batch size
image_height = 224
image_width = 224
batch_size = 32

# Use data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.15  # Split data into 85% train and 15% validation
)

# Load and split the data into training and validation sets
train_data_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    seed=42
)

validation_data_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    seed=42
)

# Create a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Optional dropout layer for regularization
    Dense(1, activation='sigmoid')  # 1 output neuron for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_generator,
    epochs=10,
    validation_data=validation_data_generator
)

# Save the trained model to a file
model.save('gender_classification_model.h5')

# Evaluate the model on the test set (if you have a separate test dataset)
test_data_generator = train_datagen.flow_from_directory(
    'path_to_test_data_directory',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary',
    seed=42
)
test_loss, test_accuracy = model.evaluate(test_data_generator)
print(f"Test accuracy: {test_accuracy}")
