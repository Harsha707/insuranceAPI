#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:18:17 2024
@author: jayvardhan
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pickle

app = FastAPI()

# Define the model input with Pydantic
class ModelInput(BaseModel):
    age: int
    sex: Literal["male", "female"]
    bmi: float
    children: Literal[0, 1, 2, 3, 4, 5]
    smoker: Literal["yes", "no"]
    region: Literal["southeast", "southwest", "northeast", "northwest"]

# Load the trained model
insurance_model = pickle.load(open('insurance_model.sav', 'rb'))

# Define the mappings for encoding
sex_encoding = {'female': 1, 'male': 0}
smoker_encoding = {'yes': 0, 'no': 1}
region_encoding = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}

@app.post('/insurance_prediction')
def insurance_pred(input_parameters: ModelInput):
    # Convert input parameters to dictionary format
    input_data = input_parameters.dict()

    # Extract and encode the input data
    age = input_data['age']
    sex = sex_encoding[input_data['sex']]
    bmi = input_data['bmi']
    children = input_data['children']
    smoker = smoker_encoding[input_data['smoker']]
    region = region_encoding[input_data['region']]

    # Create the input list for prediction
    input_list = [age, sex, bmi, children, smoker, region]

    # Make the prediction
    prediction = insurance_model.predict([input_list])

    # Return the prediction as a JSON response
    return {'prediction': prediction[0]}
