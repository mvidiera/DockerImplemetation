# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__) #app start
Swagger(app)

pickle_in = open("classifier.pkl","rb") #Open pickle file in read byte mode
classifier=pickle.load(pickle_in) # load the opened pickle file into classifier

# root API or root page which is the first thing when I get into my webpage
@app.route('/')
def welcome():
    return "Welcome All"

# creating another fucntion which is used to get all 4 features to predict the authentication of bank notes
@app.route('/predict',methods=["Get"]) #decorator which specifies the link extension and get method which is to input the values
# run in 127.0.0.1:5000/predict
# in BankNoteAuthenticationFlask.ipynb, there are 4 features
# 1. variance 2. skewness 3. curtosis 4. entropy

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
    # requesting all 4 input features and saving in varibles. 
    
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    
    # Pass all 4 features to predict() to predict the authentication 
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    # print out the result: class 0-> fake class 1-> real 
    # convert prediction into string and then return class 0 or 1. 
    return "Hello The answer is"+str(prediction)

# also: I can make changes for features in the url bar and check the prediction
#127.0.0.1:5000/predict?variance=0,&skewness=-3..... and so on for all features.

# To take entire values from a file(multiple values)
#copying same method from above and making changes in methods
#method: post is used. Because ll the values from testfile (that is from body of test file)
# explaination: POST is used to send data to a server to create/update a resource.
#The data sent to the server with POST is stored in the request body of the HTTP request
    
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
    #read csv file and store in dataframe(pandas).
    # to get csv file: request.files.get("file"). file variable has whole data test.csv
    
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test) #df_test value is passed to classfier() 
    #prediction value is printed. 
    
    return str(list(prediction)) # as there are multiple entries of features, I will convert into list()

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
    
    