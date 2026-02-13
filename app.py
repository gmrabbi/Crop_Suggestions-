import numpy as np 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model/crop_model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)[0]
    return render_template('index.html', prediction_text=f"The Predicted Crop is {prediction}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
