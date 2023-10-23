import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import sklearn
st.title('Iris Prediction App')
image = Image.open('pexels-saliha-sevim-7651066.jpg')
st.image(image)
iris_model = pickle.load( open('segment_knn.sav','rb'))
def user_report():
    sepal_length = st.sidebar.slider('sepal length',1.0,10.0,0.1)
    sepal_width = st.sidebar.slider('sepal width',1.0,10.0,0.1)
    petal_length = st.sidebar.slider('petal length',1.0,10.0,0.1)
    petal_width = st.sidebar.slider('petal width',1.0,10.0,0.1)
    data_report = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    data = pd.DataFrame(data_report, index=[0])
    return data
user_data = user_report()
st.write(user_data)
prediction = iris_model.predict(user_data)
st.success(prediction)
if (prediction ==0):
    st.success
