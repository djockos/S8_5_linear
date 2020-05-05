from flask import Flask, render_template, redirect, url_for, request, make_response, jsonify
from sklearn.externals import joblib
import numpy as np
import requests
import json

import pandas as pd

app = Flask(__name__)


# Chargement des fichiers avec les objets sklearn pour le preprocessing et le modèle
imputer = joblib.load("linear_classification_imputer.pkl")
featureencoder = joblib.load("linear_classification_featureencoder.pkl")
labelencoder = joblib.load("linear_classification_labelencoder.pkl")
classifier = joblib.load("linear_classification_classification_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        # Recover informations from html form
        data = dict(request.form.items())

        country = data["Country"]
        # handling missing fields for age and salary
        try:
            age = float(data["Age"])
        except ValueError:
            age = None
        try:
            salary = float(data["Salary"])
        except ValueError:
            salary = None

        # Create DataFrame with columns in the same order as in src/Data.csv
        x_test = {'Country':country,'Age':age,'Salary':salary}
        df = pd.DataFrame(data=x_test,index=[0])

        # Convert dataframe to numpy array before using scikit-learn
        df = df.values
        # Preprocessings : impute and scale/encode features
        df[:,[1,2]] = imputer.transform(df[:,[1,2]])
        df = featureencoder.transform(df)

        # Prediction
        y_pred = model.predict(df)

        # Use labelencoder to translate prediction into 'yes' or 'no'
        y_pred = labelencoder.inverse_transform(y_pred)
        prediction_translated = str(y_pred)

    return render_template("predicted.html", text=prediction_translated)


if __name__ == '__main__':
    app.run(debug=True)
