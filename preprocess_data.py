import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Dataset path
DATASET_PATH = "datasets/archive"  # Update this to point to the 'archive' folder

# Emotion labels mapping (you may need to adapt this if emotions are encoded differently)
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Extract features from audio
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Load dataset and extract features
features, labels = [], []

# Check if the dataset path exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path does not exist: {DATASET_PATH}")

# Traverse all subdirectories to find .wav files
for root, dirs, files in os.walk(DATASET_PATH):  # This will traverse subfolders (actors)
    for file in files:
        if file.endswith(".wav"):
            print(f"Processing file: {file}")  # Debugging line
            try:
                # Assuming emotion is in the filename and follows the format:
                # '01-01-01-01-01-01-01.wav' where the emotion code is the 3rd part
                # Example: "01" = neutral, "02" = calm, etc.
                label_code = file.split("-")[2]
                if label_code in emotions:
                    label = emotions[label_code]
                    features.append(extract_features(os.path.join(root, file)))
                    labels.append(label)
                else:
                    print(f"Skipping file with unknown emotion: {file}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

# Check if data was processed
if not features or not labels:
    raise ValueError("No audio files were processed. Check the dataset path or file format.")

# Encode labels
label_map = {emotion: idx for idx, emotion in enumerate(emotions.values())}
encoded_labels = [label_map[label] for label in labels]

# Convert to numpy arrays
X = np.array(features)
y = to_categorical(np.array(encoded_labels))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save data for training
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Preprocessing completed successfully!")
