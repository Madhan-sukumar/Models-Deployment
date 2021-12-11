# -*- coding: utf-8 -*-

import numpy as np
import pickle

#loading the saved model
load_model = pickle.load(open('E:/jupyterfiles/Practice/Ml Models/Gold Price Prediction/Gold_prediction_model.pkl','rb'))

input_data = (683.380005,27.99,13.18,1.263807)

#converting list of values into array
input_array  = np.asarray(input_data)

#reshaping the array
reshaped_array = input_array.reshape(1,-1)

#predicting the input either it is rock or mine
predicted_value = load_model.predict(reshaped_array)

print(predicted_value)

