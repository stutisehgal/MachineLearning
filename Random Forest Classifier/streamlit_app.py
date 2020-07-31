# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:49:34 2020

@author: Lenovo y540 bnin
"""

import pandas as pd
import numpy as np
import pandas as pd
import pickle
import streamlit as st

from PIL import Image

pickle_in=open('classifier.pkl','rb') #read-byte mode
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome all"

def predict_note_authentication(variance,skewness,curtosis,entropy):
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    #predict() requires 2-d matrix hence double [] braces
    print(prediction)
    return prediction

def main():
    st.title("Bank Authenticator")
    html_temp="""
    #streamlit can integrate static HTML code
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    variance=st.text_input("Variance","")
    skewness=st.text_input("Skewness","")
    curtosis=st.text_input("Curtosis","")
    entropy=st.text_input("Entropy","")
    result="" 
    #empty variable--will be updated to 0 or 1 
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")
        
if __name__=='__main__':
    main()