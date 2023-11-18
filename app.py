import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from ml_model import load_pkl

app = Flask(__name__)
model = load_pkl(path='regmodel.pkl')

@app.route('/') # root url
def home():
    return render_template('home.html')

@app.route("/predict_api", methods=["POST"]) # when the url is /predict_api run the function below
def predict_api():
    data = request.json['data']  # Corrected method to get JSON data
    prediction = model.predict(data)  # Assuming the data is in {'features': [feature_values]} format
    return jsonify(prediction.tolist())  # Converting NumPy array to list for JSON serialization

if __name__ == "__main__":
    app.run(debug=True)