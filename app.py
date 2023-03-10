import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import os

app=Flask(__name__)     # starting point of our app

IMG_FOLDER=os.path.join('static','IMG')
app.config['UPLOAD_FOLDER']=IMG_FOLDER

##Load the model
regmodel=pickle.load(open('Housing_reg_model.pkl', 'rb')) #rb

@app.route('/')
def home():
    return render_template("index.html")       

@app.route('/predict', methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()]
    data = np.array(data)
    print(x)
    prediction = model.predict(x)
    print(prediction)
    image=prediction[0]+'.png'
    image=os.path.join(app.config['UPLOAD_FOLDER'],image)

   
    return render_template('index.html',prediction=prediction[0],image=image)
                   
    
if __name__=="__main__":
    app.run(debug=True)