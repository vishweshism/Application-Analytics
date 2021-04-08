import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from Cleanfunc import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    year = request.form.get("year")
    manufacturer = request.form.get("manufacturer")
    model = request.form.get("model")
    condition = request.form.get("condition")
    cylinders = request.form.get("cylinders")
    fuel = request.form.get("fuel")
    model = request.form.get("model")
    odometer = request.form.get("odometer")
    output = prodClean(year, manufacturer, condition, fuel, odometer)
    
    return render_template('index.html', prediction_text='Car Price Is Estimated to be: $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)