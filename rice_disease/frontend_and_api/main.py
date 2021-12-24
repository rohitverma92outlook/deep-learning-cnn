import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras

st.title("Prediction of Rice Crop Disease")

data = st.file_uploader("Please upload a Rice crop image",type=['png','jpeg','jpg'])
uploaded_image = Image.open(data)
st.image(uploaded_image, caption='Uploaded Rice crop photo')

model = tf.keras.models.load_model("../saved_models/version_1")
class_names = ['Rice Brown Spot', 'Rice Healthy', 'Rice Hispa', 'Rice Leaf Blast']

image = np.array(uploaded_image)
image = tf.image.resize(image,[128,128])
img_batch = np.expand_dims(image, 0)
predictions = model.predict(img_batch)
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0])

st.subheader("Artificial Intelligence Model Prediction")
st.write(f"Predicted Category is {predicted_class} with {confidence*100}% accuracy")
