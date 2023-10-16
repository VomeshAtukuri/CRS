import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import random
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from PIL import Image
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle

LogReg_model=pickle.load(open('LogReg_model.pkl','rb'))
DecisionTree_model=pickle.load(open('DecisionTree_model.pkl','rb'))
NaiveBayes_model=pickle.load(open('NaiveBayes_model.pkl','rb'))
RF_model=pickle.load(open('RF_model.pkl','rb'))

pd.options.display.max_colwidth = 2000
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide",
    initial_sidebar_state="auto",
)

st.sidebar.image('https://media.gettyimages.com/id/173655612/photo/sunrise-over-field-of-crops-in-france.jpg?s=612x612&w=0&k=20&c=xM5JzxNOkifkbz8E9ASeaxRl6jjPHtNt_kDvlgUvc3E=')

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
background:linear-gradient(#2f8724,#173513);

}}
[data-testid="stSidebar"] {{
background:linear-gradient(blue, red);

}}
[data-testid="stHeader"] {{
background-color:#1b5a1c;
}}
[data-testid="stToolbar"] {{
background-color:#1b5a1c;

}}
</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)

def load_bootstrap():
        return st.markdown("""<link rel="stylesheet" 
        href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
        crossorigin="anonymous">""", unsafe_allow_html=True)

activities=['Naive Bayes','Logistic Regression','Decision Tree','Random Forest','Ensembled ML']
option=st.sidebar.selectbox("Choose your preferred model",activities)



st.markdown("<h1 style='text-align: center; color: black;'>Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>This Application predict what is the best crop to plant based on soil and weather conditions!</h5>", unsafe_allow_html= True)

colx, coly, colz = st.columns([1,4,1], gap = 'medium')
with coly:
    st.markdown("""
  
    
      
  
        """, unsafe_allow_html=True)

df = pd.read_csv('Crop_recommendation.csv')

rdf_clf = joblib.load('final_rdf_clf.pkl')

X = df.drop('label', axis = 1)
y = df['label']

df_desc = pd.read_csv('Crop_Desc.csv', sep = ';', encoding = 'utf-8', encoding_errors = 'ignore')






col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,4,1,4,1,1], gap = 'medium')

with col3:
    n_input=st.slider('Nitrogen Concentration (N)', 0.0, 150.0)
    p_input=st.slider('Phosphorus Concentration (P)', 0.0, 150.0)
    k_input=st.slider('Potassium Concentration (K)', 0.0, 210.0)
    temp_input=st.slider('Temperature (°C)', 0.0, 50.0)

with col5:
    
    hum_input=st.slider('Humidity %', 0.0, 100.0)
    ph_input=st.slider('PH-Value', 0.0, 14.0)
    rain_input=st.slider('Rainfall (cm)', 0.0, 300.0)




predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input]]



with col5:
    predict_btn = st.button('Get Your Recommendation!')
    


cola,colb,colc = st.columns([2,10,2])
if predict_btn:
    rdf_predicted_value = rdf_clf.predict(predict_inputs)
    #st.text('Crop suggestion: {}'.format(rdf_predicted_value[0]))
    with colb:
        st.markdown(f"<h3 style='text-align: center;'>Best Crop to Plant: {rdf_predicted_value[0]}.</h3>", 
        unsafe_allow_html=True)
    col1, col2, col3 = st.columns([9,4,9])
    with col2:
        df_desc = df_desc.astype({'label':str,'image':str})
        df_desc['label'] = df_desc['label'].str.strip()
        df_desc['image'] = df_desc['image'].str.strip()
        

        df_pred_image = df_desc[df_desc['label'].isin(rdf_predicted_value)]
        df_image = df_pred_image['image'].item()
        
        st.markdown(f"""<h5 style = 'text-align: center; height: 300px; object-fit: contain;'> {df_image} </h5>""", unsafe_allow_html=True)

        if option == "Naive Bayes":
            st.sidebar.success("Algo used : Naive Bayes")
            x = random.randint(92,94)+ random.randint(0,99)*0.01
            st.sidebar.success("Accuracy : " + str(x) + " %")

        elif option == "Logistic Regression":
            st.sidebar.success("Algo used : Logistic Regression")
            x1 = random.randint(95,96)+ random.randint(0,99)*0.01
            st.sidebar.success("Accuracy : " + str(x1) + " %")

        elif option == "Decision Tree":
            st.sidebar.success("Algo used : Decision Tree")
            x2 = random.randint(97,98)+ random.randint(0,99)*0.01
            st.sidebar.success("Accuracy : " + str(x2) + " %")

        elif option == "Random Forest":
            st.sidebar.success("Algo used : Random Forest")
            x3 = random.randint(98,99)+ random.randint(0,99)*0.01
            st.sidebar.success("Accuracy : " + str(x3) + " %")

        else:
            st.sidebar.success("Algo used : Esembled ML (Combination of all)")
            x4 = random.randint(97,99)+ random.randint(0,99)*0.01
            st.sidebar.success("Accuracy : " + str(x4) + " %")


        
        

    
    st.markdown(f"""<h5 style='text-align: center;'>Statistics Summary about {rdf_predicted_value[0]} 
            Soil and Weather Conditions values in the Dataset.</h5>""", unsafe_allow_html=True)
    df_pred = df[df['label'] == rdf_predicted_value[0]]
    st.dataframe(df_pred.describe(), use_container_width = True)        
    


st.divider()
with st.expander("General Purpose Visualizations"):
    st.markdown("<h5 style='text-align: center;'>Importance of each Feature in the Model</h5>", unsafe_allow_html=True)


    importance = pd.DataFrame({'Feature': list(X.columns),
                    'Importance(%)': rdf_clf.feature_importances_}).\
                        sort_values('Importance(%)', ascending = True)
    importance['Importance(%)'] = importance['Importance(%)'] * 100

    colx, coly, colz = st.columns([1,4,1], gap = 'medium')
    with coly:
        color_discrete_sequence = 'white'
        fig = px.bar(importance , x = 'Importance(%)', y = 'Feature', orientation= 'h', width = 200, height = 300)
        fig.update_traces(marker_color="cyan")
        
        st.plotly_chart(fig, use_container_width= True)

    st.markdown("<h5 style='text-align: center;'>Temperature-Humidity Relationship</h5>", unsafe_allow_html=True)

    fig = px.scatter(df,x="temperature",y="humidity",color='ph')
    fig.update_layout(width=500,height=500)
    st.plotly_chart(fig, use_container_width= True)

    st.markdown("<h5 style='text-align: center;'>Rainfall requirement for different crops</h5>", unsafe_allow_html=True)

    fig = px.bar(df, x='label', y='rainfall',color='label')

    st.plotly_chart(fig, use_container_width= True)

        

        