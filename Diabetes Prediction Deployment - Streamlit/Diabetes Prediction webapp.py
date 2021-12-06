# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:04:10 2021

@author: Madhan
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
load_model = pickle.load(open('E:/jupyterfiles/Practice/Ml Models/Diabetes Prediction for Women/Diabetes_Prediction_model.pkl','rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
    
    #convert the input data into array
    array_data = np.asarray(input_data)

    #reshaping the converted array data
    reshaped_data = array_data.reshape(1,-1)


    #predicting whether having diabetes or not
    prediction = load_model.predict(reshaped_data)

    print(prediction)

    if(prediction[0]== 0):
        return "The Patient doesn't having Diabetes"    
    else:
        return "The Patient is Diabetic"
    
    
    
def main():
    
    # title for the Predictor
    st.title('Diabetes Prediction for Women')
    
    #getting the input data from user
    
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('Blood Pressure level')
    SkinThickness=st.text_input('SkinThickness value')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('Body Mass Index')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree value')
    Age=st.text_input('Age of Person')
    
     # code for Prediction
    #the prediction output stores finally in this variable
    diagnosis = ''
  
    #creating a button for prediction
    # after button pressed, the input values passed as key input to the function
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    
        
    
    # for displaying success message
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()
    