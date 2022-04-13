

import numpy as np 
import pickle 
import pandas as pd 
from flask import Flask, request
from flask import Flask, request, jsonify, render_template
from final_project import fuzzify


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods =['POST'])
def prediction():
    
    int_features = [x for x in request.form.values()]
    # Extracting features provided by user
    en_critic = int_features[0]
    en_globalsales = int_features[1]
    en_userrating = int_features[2]
    en_year = int_features[3]
    
    output = fuzzify(en_critic,en_globalsales,en_userrating,en_year)
    print(output,"******************")
    return render_template('index.html', prediction_text= output)



if __name__ == '__main__':
    app.run()