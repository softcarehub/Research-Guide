import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import load_model
import streamlit as st 
import numpy as np 
# import pandas as pd
# from PIL import Image
model = load_model('C:\\Users\\bariu\\Python\\Image_Classification\\Image_classify.keras')

data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

img_height =180
img_width =180

# Editing
# st.image('corn.jpg')
# Set a nice title

# st.title("Check Fruits and Vegetables")

# Set a default value with the ".jpg" extension
default_value = data_cat[35] + '.jpg'

# Create a text input with the default value
image = st.text_input('Enter Image Name:', default_value)

# Display the input
st.write(f'You entered: {image}')

#Editing

# image= st.text_input('Enter Image Name:','watermelon.jpg')

image_load = tf.keras.utils.load_img(image,target_size= (img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image,width=400)

st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)]) 
st.write('With Accuracy of ' + str(np.max(score)*100))
st.markdown('###  `Build By Bariul`') 
# For run step-1:  Run python file in dedicated terminal
# For run step-2:  python -m streamlit run app.py  
# Learn streamlit: https://docs.streamlit.io/library/api-reference/write-magic/st.write