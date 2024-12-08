from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the emotion and genre detection models
emotion_model = load_model('models/emotion_model.h5')
genre_model = joblib.load('models/genre_model.pkl')

# Emotion and genre labels
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
genre_labels = ['blues', 'pop', 'rock', 'classical', 'jazz', 'metal', 'country', 'disco', 'hiphop', 'reggae']

# Initialize the LabelEncoder for both emotion and genre
emotion_label_encoder = LabelEncoder()
emotion_label_encoder.fit(emotion_labels)

genre_label_encoder = LabelEncoder()
genre_label_encoder.fit(genre_labels)

# Log the genre labels to verify encoding
logging.debug(f"Genre labels: {genre_labels}")
logging.debug(f"Encoded genre labels: {genre_label_encoder.transform(genre_labels)}")

def extract_features(file_path):
    """Extract MFCC features from the audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=2.5, offset=0.6)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {e}")
        raise

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    """Predict emotion from the uploaded audio file."""
    try:
        # Get the audio file from the form
        file = request.files['audio']
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Extract features and make prediction
        features = extract_features(file_path)
        features = np.array([features])
        prediction = emotion_model.predict(features)
        predicted_emotion = emotion_label_encoder.inverse_transform([np.argmax(prediction)])

        return jsonify({"emotion": predicted_emotion[0]})
    except Exception as e:
        logging.error(f"Error in emotion prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_genre', methods=['POST'])
def predict_genre():
    """Predict genre from the uploaded audio file."""
    try:
        # Get the audio file from the form
        file = request.files['audio']
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Extract features and make prediction
        features = extract_features(file_path)
        features = np.array([features])

        # Log the extracted features for debugging
        logging.debug(f"Extracted features: {features}")

        prediction = genre_model.predict(features)
        logging.debug(f"Raw prediction output: {prediction}")

        # If the model gives probabilities, use the highest value
        if isinstance(prediction[0], np.ndarray):  # probabilities (e.g., from softmax)
            predicted_genre_index = np.argmax(prediction[0])
        else:  # class indices
            predicted_genre_index = prediction[0]

        # Decode the genre label using the LabelEncoder
        predicted_genre = genre_label_encoder.inverse_transform([predicted_genre_index])
        logging.debug(f"Predicted genre: {predicted_genre[0]}")

        return jsonify({"genre": predicted_genre[0]})
    except Exception as e:
        logging.error(f"Error in genre prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
