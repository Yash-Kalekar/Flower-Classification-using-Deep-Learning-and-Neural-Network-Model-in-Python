import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('Flower_Recog_Model.h5')

def classify_images(image):
    input_image = image.resize((180, 180)) 
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, width=200)


    result = classify_images(image)
    st.write(result)
