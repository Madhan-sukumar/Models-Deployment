# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:56:39 2021

@author: Madhan
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st 



load_model=pickle.load(open('E:/jupyterfiles/Practice/Ml Models/Bank note authentication/bank_authentication_classifier.pkl','rb'))


def predict_note_authentication(variance,skewness,curtosis,entropy):
       
    prediction=load_model.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction



def main():
    st.title("Bank Authenticator Application")
    html_temp = """
    <div style="background-color:orange;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.text_input("Variance","Type Here")
    skewness = st.text_input("skewness","Type Here")
    curtosis = st.text_input("curtosis","Type Here")
    entropy = st.text_input("entropy","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success('The output is {}'.format(result))


if __name__=='__main__':
    main()