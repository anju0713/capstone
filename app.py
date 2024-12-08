import streamlit as st
import requests
import numpy as np
import os
import librosa
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import pydub
from pydub.playback import play
from io import BytesIO

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

def extract_features(file_path):
    """Extract MFCC features from the audio file."""
    audio, sr = librosa.load(file_path, sr=22050, duration=2.5, offset=0.6)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Streamlit app layout
st.title("Emotion and Genre Detection")

# Emotion Detection Section
st.header("Emotion Detection")
uploaded_emotion_file = st.file_uploader("Upload an audio file for emotion detection", type=["wav", "mp3", "ogg"])

if uploaded_emotion_file is not None:
    st.audio(uploaded_emotion_file, format='audio/wav')
    if st.button("Predict Emotion"):
        file_path = os.path.join('static', uploaded_emotion_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_emotion_file.getbuffer())

        features = extract_features(file_path)
        features = np.array([features])
        prediction = emotion_model.predict(features)
        predicted_emotion = emotion_label_encoder.inverse_transform([np.argmax(prediction)])
        st.write("Predicted Emotion: ", predicted_emotion[0])

# Genre Detection Section
st.header("Genre Detection")
uploaded_genre_file = st.file_uploader("Upload an audio file for genre detection", type=["wav", "mp3", "ogg"])

if uploaded_genre_file is not None:
    st.audio(uploaded_genre_file, format='audio/wav')
    if st.button("Predict Genre"):
        file_path = os.path.join('static', uploaded_genre_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_genre_file.getbuffer())

        features = extract_features(file_path)
        features = np.array([features])
        prediction = genre_model.predict(features)

        if isinstance(prediction[0], np.ndarray):  # probabilities (e.g., from softmax)
            predicted_genre_index = np.argmax(prediction[0])
        else:  # class indices
            predicted_genre_index = prediction[0]

        predicted_genre = genre_label_encoder.inverse_transform([predicted_genre_index])
        st.write("Predicted Genre: ", predicted_genre[0])

# Audio Recording for Emotion Detection
st.header("Record Audio for Emotion Detection")
webrtc_ctx = webrtc_streamer(
    key="emotion-recording",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    ),
)

if webrtc_ctx.state.playing:
    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            audio = np.concatenate([frame.to_ndarray() for frame in audio_frames])
            audio_segment = pydub.AudioSegment(
                audio.tobytes(), frame_rate=webrtc_ctx.audio_receiver.sample_rate, sample_width=2, channels=1
            )
            play(audio_segment)
            st.audio(audio_segment.raw_data, format='audio/wav')

            # Save the audio to a file
            audio_file = BytesIO()
            audio_segment.export(audio_file, format="wav")
            audio_file.seek(0)

            files = {"audio": audio_file}
            response = requests.post("http://localhost:5000/predict_emotion", files=files)
            if response.status_code == 200:
                st.write("Predicted Emotion: ", response.json()["emotion"])
            else:
                st.write("Error in prediction")
