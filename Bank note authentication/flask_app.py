

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

#loading the saved model
load_model = pickle.load(open("E:/jupyterfiles/Practice/Ml Models/Bank note authentication/bank_authentication_classifier.pkl","rb"))

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"]) 
def predict_note_authentication():
    
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
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=load_model.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "The prediction is"+ str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=load_model.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='127.0.0.1',port=8000)
    
    