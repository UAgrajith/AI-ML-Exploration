import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model/digit_model.h5")

st.title("🧠 Digit Recognizer Web App")
st.write("Upload an image of a digit (0–9)")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", width="stretch")
    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28)

    # Prediction
    prediction = model.predict(img)
    result = np.argmax(prediction)

    st.success(f"Predicted Digit: {result}")