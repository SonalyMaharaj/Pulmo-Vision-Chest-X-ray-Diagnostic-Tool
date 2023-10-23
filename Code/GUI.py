import numpy as np
import pandas as pd
import os
import streamlit as st
import matplotlib.pyplot as plt
import skimage
import pickle as pkl

#Load the model
Model = pkl.load(open('../Saved Models/KNN_3.pkl','rb'))

#Preprocessing Functions
def Image_Noise_Reduction(image):
    '''Function to reduce noise in the image'''
    blurred_image = skimage.filters.gaussian(image, sigma=0.5)
    return blurred_image

def Image_Enhance_Contrats(image):
    '''Function to enhance the contrast of the image'''
    enhanced_image = skimage.exposure.equalize_hist(image)
    return enhanced_image

def Image_Normalize(image):
    '''Function to normalize the image'''
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return normalized_image

def Image_Preprocessing(image):
    '''Function to preprocess the image'''
    blurred_image = Image_Noise_Reduction(image)
    enhanced_image = Image_Enhance_Contrats(blurred_image)
    normalized_image = Image_Normalize(enhanced_image)
    return np.array(normalized_image)


st.title('Pulmo-Vision')

with st.container():
    uploaded_Image = st.file_uploader("Upload an image for diagnosis")

    col1, col2, col3 , col4, col5 = st.columns(5)
    buttong_click = None
    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        buttong_click = st.button("Diagnose")

st.divider()

if buttong_click:
    if uploaded_Image is not None:
        Uploaded_Image = np.array(skimage.transform.resize(skimage.io.imread(uploaded_Image), (100,100)))
        Preprocessed_Image = Image_Preprocessing(Uploaded_Image) 
        Prediction,Confidence = Model.predict(Preprocessed_Image.flatten().reshape(1,-1))

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(Uploaded_Image,caption='Uploaded Image',width=250)
            with col2:
                st.image(Preprocessed_Image,caption='Preprocessed Image',width=250)

        with st.container():        
            st.write(f'Diagnostic Result: **{Prediction[0]}**')
            st.write(f'Confidence Level: **{Confidence[0]*100}%**')
