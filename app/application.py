# Import library
import os
import pandas as pd
import pickle
import numpy as np
from flask import redirect, render_template, url_for, request, json
from app import app

# app.config
rf_model = pickle.load(open('app/models/model.pkl', 'rb'))

# Render landing page
@app.route('/')
def index():
    return render_template('index.html')

# Render predict page
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Render predict page
@app.route('/result', methods=['POST'])
def result_predict():
    if request.method == 'POST':
        volatile_acidity = float(request.form['VolatileAcidity'])
        citric_acid = float(request.form['CitricAcid'])
        chlorides = float(request.form['Chlorides'])
        total_sulfur_dioxide = float(request.form['TotalSulfurDioxide'])
        sulphates = float(request.form['Sulphates'])
        alcohol = float(request.form['Alcohol'])
        
        input_data = (volatile_acidity, citric_acid, chlorides,
        total_sulfur_dioxide, sulphates, alcohol)

        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaphed = input_data_as_np_array.reshape(1,-1)
        prediction = rf_model.predict(input_data_reshaphed)

        if (prediction[0] == 1):
            result = 'Good Quality'
            url = "https://em-content.zobj.net/source/microsoft-teams/337/winking-face_1f609.png"
        else:
            result = 'Sorry, your wine is Bad Quality'
            url = "https://em-content.zobj.net/source/microsoft-teams/337/worried-face_1f61f.png"

        return render_template('result.html', result=result, url_img=url)
    else:
        return redirect(url_for('predict'))