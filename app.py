from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load('trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info('Received prediction request')
    try:
        data = request.get_json()
        app.logger.info(f'Received data: {data}')
        
        # Extract features and convert them to float
        features = [float(data['feature1']), float(data['feature2']), float(data['feature3']), float(data['feature4'])]
        
        # Make prediction
        prediction = model.predict([features])
        
        app.logger.info(f'Prediction: {prediction}')
        return jsonify({'prediction': prediction.tolist()}), 200
    except Exception as e:
        app.logger.error(f'Error processing prediction request: {str(e)}')
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
