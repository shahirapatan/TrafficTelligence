from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)


# Load pre-trained models and encoders
model_path = "best_model.pkl"
scaler_path = "scale.pkl"
encoder_path = "encoder.pkl"
transformer_path = "scaler.pkl"


model = pickle.load(open(model_path, 'rb'))
##scaler = pickle.load(open(scaler_path, 'rb'))
##encoder = pickle.load(open(encoder_path, 'rb'))
transformer = pickle.load(open(transformer_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    if request.method == 'POST':
        # Example: Collecting input fields
        # holiday = int(request.form['holiday'])
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = float(request.form['weather'])
        year = request.form['year']
        month = request.form['month']
        day = request.form['day']
        hours = request.form['hours']
        minutes = request.form['minutes']
        seconds = request.form['seconds']

        # Combine inputs into a single array for prediction
        features = np.array([[ temp, rain, snow, day, month, year, hours, minutes, seconds, weather]])
        

        
        print(features)
        
        transformed_features = transformer.transform(features)

        # Predict
        prediction = model.predict(transformed_features)

        # Example: Rendering results
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)