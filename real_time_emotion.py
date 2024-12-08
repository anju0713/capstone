import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pyaudio

# Load the pre-trained emotion model
model = load_model('emotion_model.h5')

# Define emotion labels
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Function to predict emotion from audio
def predict_emotion(audio_clip):
    # Extract MFCC features from audio
    mfcc = librosa.feature.mfcc(y=audio_clip, sr=22050, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)  # Average across time
    prediction = model.predict(np.expand_dims(mfcc, axis=0))  # Predict emotion
    return emotions[np.argmax(prediction)]

# PyAudio setup for real-time input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=22050, input=True, frames_per_buffer=1024)

print("Listening for emotions...")

# Main loop to continuously listen for audio and predict emotion
while True:
    data = np.frombuffer(stream.read(1024), dtype=np.float32)
    emotion = predict_emotion(data)
    print(f"Detected Emotion: {emotion}")
