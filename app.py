from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
X = pd.read_csv('X.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])

def predict():
    features = [x for x in request.form.values()]
    features = np.array(features)
    location = features[3]
    print(X)
    loc_index = np.where(X.columns==location)[0][0]
    if loc_index >= 0:
        features[3] = 1
    else:
        features[3] = 0
    features = features.astype(np.int64)
    features = np.append(features, np.zeros(239))
    print(features[0].dtype)
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text = "Housing price should be Rs. {} Lac".format(output))

if __name__ == "__main__":
    app.run(debug=True)

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)