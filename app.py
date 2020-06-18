import streamlit as st
from datasetPrepare.createCaptcha import create_captcha
from datasetPrepare.sliceImg import segment_image
import pandas as pd
import pickle
from TrainPredict.PredictWord import predict_captcha

URI = 'http://127.0.0.1:5000/'
st.title('Captcha Cracker')
st.sidebar.markdown('## Control Panel')

default_value = ''
user_input = st.text_input("Type in a 6 Characters Capitalized word ", default_value)
shear = st.slider("Shear Value", 0.01, 0.35, key='shear')
image = create_captcha(user_input, shear=shear)

if st.button('Create Captcha'):
    st.write('Success Created Captcha!')
    st.image(image=image, width=400)

if st.sidebar.button('Slice Image'):
    st.write('Success Slice Image!')
    image_slice = segment_image(image)
    st.image(image=image_slice, width=70)

if st.sidebar.checkbox("Show raw data Image", False):
    df_Image = pd.read_csv('Dataset/images.csv')
    st.subheader("Image Dataset")
    st.write(df_Image)

if st.sidebar.checkbox("Show raw data Labels", False):
    st.subheader("Image Dataset")
    df_Labels = pd.read_csv('Dataset/label.csv')
    st.write(df_Labels)

if st.sidebar.button('Make Prediction'):
    st.subheader("MLP Prediction result")
    MLP_clf = pickle.load(open('TrainPredict/MultiPerceptronModel.sav', 'rb'))
    y_pred, predictions, predict_words = predict_captcha(image, MLP_clf)
    st.write("y Prediction (neuron output)")
    st.write(y_pred)
    st.write("Argmax output")
    st.write(predictions)
    st.subheader("Final Prediction:")
    st.subheader(predict_words)
