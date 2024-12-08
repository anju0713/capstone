import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Dataset path
DATASET_PATH = "C:/datasets/archive"

# Emotion labels mapping
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
for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        label = emotions[file.split("-")[2]]
        features.append(extract_features(os.path.join(DATASET_PATH, file)))
        labels.append(label)

# Encode labels
label_map = {emotion: idx for idx, emotion in enumerate(emotions.values())}
encoded_labels = [label_map[label] for label in labels]

# Convert to numpy arrays
X = np.array(features)
y = to_categorical(np.array(encoded_labels))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
