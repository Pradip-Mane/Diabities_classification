import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import os

app=Flask(__name__)     # starting point of our app

IMG_FOLDER=os.path.join('static','IMG')
app.config['UPLOAD_FOLDER']=IMG_FOLDER

##Load the model
model=pickle.load(open('SVM_Diabities_Class_model.pkl', 'rb')) #rb

@app.route('/')
def home():
    return render_template("index.html")       

@app.route('/predict', methods=['POST'])
def predict():

    data1=[float(x) for x in request.form.values()]
    data = np.array([data1])
    prediction = model.predict(data)
    image=str(prediction[0])+'.png'
    image=os.path.join(app.config['UPLOAD_FOLDER'],image)

   
    return render_template('index.html',prediction=prediction[0],image=image)
                   
    
if __name__=="__main__":
    app.run(debug=True)