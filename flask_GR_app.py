# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 23:42:33 2021

@author: Poojan
"""

import numpy as np 
import pickle 
import pandas as pd 
from flask import Flask, request
from flask import Flask, request, jsonify, render_template
from Knowledge_and_Expert_systems_final_project import fuzzify


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods =['POST'])
def prediction():
    
    int_features = [x for x in request.form.values()]
    # Extracting features provided by user
    ip_critic = int_features[0]
    ip_userrating = int_features[1]
    ip_globalsales = int_features[2]
    ip_year = int_features[3]
    
    # Using the method from main file to get output
    result = fuzzify(ip_critic,ip_userrating,ip_globalsales,ip_year)
    print(result,"******************")
    return render_template('index.html', prediction_text= result)



if __name__ == '__main__':
    app.run()