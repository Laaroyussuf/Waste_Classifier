import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import img_to_array 

model = load_model('model_final.h5')

def sort_waste(image):
  if image is None:
      return st.write("Please upload an image")
    
  st.write("Image received successfully.")
  image = Image.open(image)
  image = np.array(image)
  img = cv2.resize(image, (224, 224))
  img_array = img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0
  pred = model.predict(img_array)
  pred = pred.argmax()
  return pred



st.title("Waste Sorting Model")

# Prompt user to upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "gif", "bmp", "webp", "jfif"])
if uploaded_image is not None:
  image = Image.open(uploaded_image)
  st.image(image, caption="Uploaded Image", use_column_width=True)
else:
   st.write('Please Upload an image.')

if st.button('Sort Image'):
  classifier = sort_waste(uploaded_image)
  if classifier == 1:
    st.text('The waste is Recyclable')
  else:
    st.text('The waste is Organic')