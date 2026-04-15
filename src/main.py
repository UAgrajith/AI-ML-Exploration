import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)

# Save model
model.save("digit_model.h5")

# Test with OpenCV image (optional)
img = cv2.imread('test.png', 0)
if img is not None:
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    print("Predicted number:", np.argmax(prediction))
    plt.imshow(x_test[0], cmap='gray')
    plt.title(f"Label: {y_test[0]}")
    plt.show()