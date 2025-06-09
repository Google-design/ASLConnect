from flask import Flask, request, jsonify
from ProcessFrames import process_frame
import logging
from flask_cors import CORS

app = Flask(__name__)

CORS(app, origins=["http://localhost:4200/"], supports_credentials=True, methods=['GET', 'POST', 'OPTIONS'], allow_headers=["Content-Type", "Authorization", "X-Requested-With"])
logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        timestamp = data['timestamp']
        logging.debug("Image data received")

        prediction = process_frame(image_data, timestamp)
        logging.debug(f"Prediction: {prediction}")
        return jsonify(prediction=prediction)
    except Exception as e:
        logging.error(f"Error processing image: {e}", exc_info=True)  # Log stack trace
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)
