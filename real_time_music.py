# real_time_music.py

import pickle
import librosa
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load the trained model
with open('models/genre_model.pkl', 'rb') as f:
    genre_model = pickle.load(f)

# Function to extract MFCC features from an audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)  # Load audio file in mono
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract MFCC
        return np.mean(mfcc.T, axis=0)  # Return the mean of MFCC over time
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

@app.route('/', methods=['POST'])
def predict():
    # Ensure an audio file is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded file temporarily
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Extract features from the uploaded audio file
        features = extract_features(file_path)

        if features is not None:
            # Predict the genre
            genre_prediction = genre_model.predict([features])
            return jsonify({'genre': genre_prediction[0]})
        else:
            return jsonify({'error': 'Error extracting features from the file'}), 500

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
