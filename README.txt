Fashion MNIST CNN

This project demonstrates how to build and evaluate a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the Fashion MNIST dataset. 
The Fashion MNIST dataset consists of 60,000 grayscale images of size 28x28 pixels for training and 10,000 images for testing, each labeled with one of 10 clothing categories.

Requirements
To run this code, we needed to have the following Python packages installed:

tensorflow
numpy


Dataset
The dataset used is the Fashion MNIST dataset, which is available in TensorFlow's dataset library.

Code Overview

1. Import Libraries

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from tensorflow.keras.utils import to_categorical

2. Load and Preprocess Data

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess the data
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

# Convert labels to one-hot encoding

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

3. Define the Model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
4. Compile the Model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
5. Train the Model

model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

6. Evaluate the Model

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)

7. Make Predictions

# Make predictions on two images
predictions = model.predict(test_images[:2])
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted classes for the first two images:", predicted_classes)

# Map predicted classes to labels
labels = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
          5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

print("Labels for the first two images:", labels[predicted_classes[0]], labels[predicted_classes[1]])

Results
The code trains a CNN on the Fashion MNIST dataset and evaluates its performance. It prints out the test accuracy and the predicted class labels for two test images.