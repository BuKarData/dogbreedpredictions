import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model


#importing model 
model = load_model('dog_breed.h5')

CLASS_NAMES = ['Scottish Deerhound', 'Maltese dog', 'Bernese Mountain Dog']

st.title("Dog breeds predictions")
st.markdown("Upload your dogs image")

#uploading images to app
dog_image = st.file_uploader('Choose an image: ', type = 'png')
submit = st.button('Predict')

if submit: 

    if dog_image != None:

        #converting file to cv2
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        #displaying original image
        st.image(opencv_image, channels='BGR')
        opencv_image = cv2.resize(opencv_image, (224, 224))

        #dimensionizing image
        opencv_image.shape = (1,224,224,3)
        
        Y_pred=  model.predict(opencv_image)

        st.title(str("The dog's breed is:")+CLASS_NAMES[np.argmax(Y_pred)])
