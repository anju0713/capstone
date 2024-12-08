import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
import io

# Path to the genres_original folder where subfolders represent different genres
data_path = 'datasets/data/genres_original'  # Update this if needed

# Define genre labels (the folder names inside genres_original)
genres = os.listdir(data_path)  # Get all genre folder names

# Prepare a list to hold features and labels
X, y = [], []


# Function to extract features (MFCC) from audio files using librosa
def extract_features(file_path):
    try:
        # First, try loading the audio using librosa
        y, sr = librosa.load(file_path, sr=22050, mono=True)  # Load audio file in mono
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract MFCC
        return np.mean(mfcc.T, axis=0)  # Return the mean of MFCC over time
    except Exception as e:
        print(f"Error processing {file_path} using librosa: {e}")

        # If librosa fails, try using pydub and then librosa for processing
        try:
            # Use pydub to read the wav file
            audio = AudioSegment.from_wav(file_path)
            audio = audio.set_channels(1).set_frame_rate(22050)  # Ensure mono and 22050 Hz sample rate
            wav_data = io.BytesIO()
            audio.export(wav_data, format="wav")
            wav_data.seek(0)
            y, sr = librosa.load(wav_data, sr=22050, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            return np.mean(mfcc.T, axis=0)
        except Exception as e:
            print(f"Error processing {file_path} using pydub: {e}")
            return None  # Return None if both methods fail


# Loop through each genre folder
for genre in genres:
    genre_path = os.path.join(data_path, genre)
    if not os.path.exists(genre_path):
        print(f"Genre folder {genre} does not exist, skipping.")
        continue  # Skip if the genre folder doesn't exist

    # Loop through all audio files in the genre folder
    for filename in os.listdir(genre_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(genre_path, filename)
            print(f"Processing: {file_path}")  # Log the current file being processed
            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(genre)  # Append genre label

# Convert features and labels into numpy arrays
X = np.array(X)
y = np.array(y)

# Optionally, you can save the features and labels as .npy files for later use
np.save('X_music.npy', X)
np.save('y_music.npy', y)

print("Preprocessing completed successfully!")
X = np.load('X_music.npy')
y = np.load('y_music.npy')

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")
