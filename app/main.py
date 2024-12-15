import tensorflow as tf
import os
import streamlit as st
import numpy as np
from PIL import Image
working_dir = os.path.dirname(os.path.abspath(__file__)) #to get absolute path where this file is running ie main.py
model_path = f'{working_dir}/trained_model/trained_fashion_detection.h5'
# load pre-trained model
model = tf.keras.models.load_model(model_path)
# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28,28))
    img = img.convert('L') #to greyscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1,28,28,1))
    return img_array

# Streamlit App
st.title("Fashion Item Classifier")
uploaded_image = st.file_uploader("Upload an image..." , type=['jpg' , 'jpeg' , 'png'])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1 , col2 = st.columns(2)

    with col1:
        resized_image = image.resize((100,100))
        st.image(resized_image)

    with col2:
        if st.button('Classify'):
            img_array = preprocess_image(uploaded_image)
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]
            st.success(f'Prediction: {prediction}')