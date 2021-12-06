# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:57:50 2021

@author: Madhan
"""

import numpy as np
import pickle

#loading the saved model
load_model = pickle.load(open('E:/jupyterfiles/Practice/Ml Models/Diabetes Prediction for Women/Diabetes_Prediction_model.pkl','rb'))

input_data = (4,171,72,0,0,43.6,0.479,26,)

#convert the input data into array
array_data = np.asarray(input_data)

#reshaping the converted array data
reshaped_data = array_data.reshape(1,-1)


#predicting whether having diabetes or not
prediction = load_model.predict(reshaped_data)

print(prediction)

if(prediction[0]== 0):
    print("The Patient doesn't having Diabetes")    
else:
    print("The Patient is Diabetic")