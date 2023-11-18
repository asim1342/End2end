import pickle
from flask import Flask, request, jsonify, render_template
from ml_model import load_pkl

app = Flask(__name__)
model = load_pkl(path='regmodel.pkl')

@app.route('/')  # Root URL
def home():
    return render_template('home.html')

@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        feature_value = float(request.form.get('feature'))
        prediction = model.predict([[feature_value]])  # Model expects a 2D array
        return jsonify(prediction.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
