from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np

# Initialize Flask applications
app = Flask(__name__)

model = joblib.load('./src/models/speciesPrediction.pkl')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    response = {
        'prediction': int(prediction[0])
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')