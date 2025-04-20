import streamlit as st
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model
from deepface import DeepFace
import os
import tempfile
from PIL import Image
import webbrowser

# Emotion to Music Map
emotion_music_map = {
    'happy': 'https://open.spotify.com/track/3JvrhDOgAt6p7K8mDyZwRd',
    'sad': 'https://open.spotify.com/track/7fBv7CLKzipRk6EC6TWHOB',
    'angry': 'https://open.spotify.com/track/6rqhFgbbKwnb9MLmUQDhG6',
    'surprise': 'https://open.spotify.com/track/0QOqcNIbYNY1o5qJh32wKp',
    'fear': 'https://open.spotify.com/track/4sPMNnZGsFlvQbP1W7ZFGS',
    'disgust': 'https://open.spotify.com/track/3n3Ppam7vgaVa1iaRUc9Lp',
    'neutral': 'https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC'
}

# Build model manually since we only have weights
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 emotions in FER2013
    return model

# Load weights
model = build_model()
model.load_weights("model.h5")

# Emotion classes
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Streamlit UI
st.title("🎵 Emotion-Based Music Recommender")
st.write("Upload an image or take a webcam snapshot, and get a music suggestion based on your emotion!")

# Upload or webcam
option = st.radio("Choose input method", ['Upload Image', 'Webcam'])

def predict_emotion(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    prediction = model.predict(reshaped)
    emotion_index = int(np.argmax(prediction))
    return emotion_labels[emotion_index]

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        st.image(img_np, caption='Uploaded Image', use_column_width=True)

        try:
            face_analysis = DeepFace.analyze(img_np, actions=['emotion'], enforce_detection=False)
            emotion = face_analysis[0]['dominant_emotion']
            st.success(f"Detected Emotion: **{emotion.capitalize()}**")
            music_link = emotion_music_map.get(emotion.lower(), None)
            if music_link:
                st.markdown(f"[▶️ Play Recommended Music]({music_link})", unsafe_allow_html=True)
            else:
                st.info("No music found for this emotion.")
        except Exception as e:
            st.error(f"Face or emotion not detected: {str(e)}")

elif option == 'Webcam':
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)
        img_np = np.array(image)

        st.image(img_np, caption='Captured Image', use_column_width=True)

        try:
            face_analysis = DeepFace.analyze(img_np, actions=['emotion'], enforce_detection=False)
            emotion = face_analysis[0]['dominant_emotion']
            st.success(f"Detected Emotion: **{emotion.capitalize()}**")
            music_link = emotion_music_map.get(emotion.lower(), None)
            if music_link:
                st.markdown(f"[▶️ Play Recommended Music]({music_link})", unsafe_allow_html=True)
            else:
                st.info("No music found for this emotion.")
        except Exception as e:
            st.error(f"Face or emotion not detected: {str(e)}")