import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
# Import necessary layers from Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from PIL import Image
import webbrowser # To potentially open links automatically

# --- Set Page Config FIRST ---
# This MUST be the first Streamlit command
st.set_page_config(layout="wide")

# --- Configuration ---
MODEL_WEIGHTS_PATH = 'model.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] # Match model output
EMOTION_MUSIC_MAP = {
    'happy': 'https://www.youtube.com/results?search_query=happy+songs',
    'sad': 'https://www.youtube.com/results?search_query=sad+songs',
    'angry': 'https://www.youtube.com/results?search_query=angry+songs',
    'surprise': 'https://www.youtube.com/results?search_query=surprise+songs',
    'fear': 'https://www.youtube.com/results?search_query=fear+songs',
    'disgust': 'https://www.youtube.com/results?search_query=disgust+songs',
    'neutral': 'hhttps://www.youtube.com/results?search_query=neutral+songs'
}
LABEL_MAP = {
    'Anger': 'angry',
    'Disgust': 'disgust',
    'Fear': 'fear',
    'Happy': 'happy',
    'Sad': 'sad',
    'Surprise': 'surprise',
    'Neutral': 'neutral'
}
MODEL_INPUT_SHAPE = (64, 64)

# --- Build Model Architecture ---
def build_model(input_shape=(MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1], 1), num_classes=7):
    model = Sequential(name="Emotion_CNN")

    # Conv + Pooling 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape, name='conv2d_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_1')) # Output: 32x32x32
    model.add(Dropout(0.25, name='dropout_1'))

    # Conv + Pooling 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_2')) # Output: 16x16x64
    model.add(Dropout(0.25, name='dropout_2'))

    # Conv + Pooling 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_3')) # Output: 8x8x128
    model.add(Dropout(0.25, name='dropout_3'))

    # Conv 4
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_4')) # Output: 8x8x128

    # !!! ADDED LIKELY MISSING POOLING LAYER !!!
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_4')) # Output: 4x4x128

    # Flatten and Fully Connected
    model.add(Flatten(name='flatten')) # Flattens 4x4x128 = 2048 features
    model.add(Dense(1024, activation='relu', name='dense_1')) # Expects 2048 input features
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model

# --- Load Model and Weights ---
# Use @st.cache_resource for efficient loading
@st.cache_resource
def load_models():
    try:
        model = build_model(num_classes=len(EMOTION_LABELS))
        model.load_weights(MODEL_WEIGHTS_PATH)
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        return model, face_cascade
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Ensure 'model.h5' contains weights matching the defined architecture and 'haarcascade_frontalface_default.xml' is present.")
        st.stop()

model, face_cascade = load_models()
# st.success("Models loaded successfully!") # Optional: can be removed or kept

# --- Functions ---
def predict_emotion(img_array):
    """Detects faces, predicts emotion for the first face found."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # Use scaleFactor=1.1 for potentially better detection, minNeighbors=5 is standard
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, img_array # Return None emotion, original image

    # Use the first detected face
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y + h, x:x + w]

    # Preprocess for the model - **Resize to match MODEL_INPUT_SHAPE**
    try:
        resized_face = cv2.resize(face_roi, MODEL_INPUT_SHAPE, interpolation=cv2.INTER_AREA)
    except Exception as e:
        st.warning(f"Could not resize face ROI: {e}")
        return None, img_array

    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1], 1)) # Add batch and channel dim

    # Predict emotion
    try:
        # Optional: Force CPU if GPU issues arise
        # with tf.device('/cpu:0'):
        #      prediction = model.predict(reshaped_face)
        prediction = model.predict(reshaped_face) # Simpler call
        emotion_index = np.argmax(prediction[0]) # Get prediction for the single batch item
        detected_emotion = EMOTION_LABELS[emotion_index]
    except IndexError: # Handle case where labels might not align
         st.error(f"Prediction index {emotion_index} out of bounds for labels (length {len(EMOTION_LABELS)}).")
         return None, img_array
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, img_array

    # Draw rectangle and label on the original image (for display)
    cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img_array, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return detected_emotion, img_array # Return emotion and image with rectangle/label


# --- Streamlit App UI ---
st.title("🎭 Emotion-Based Music Recommender 🎵")
st.write("Take a picture, let us detect your emotion, and get a music suggestion!")

# Initialize session state
if 'detected_emotion' not in st.session_state:
    st.session_state.detected_emotion = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'img_file_buffer' not in st.session_state:
    st.session_state.img_file_buffer = None

col1, col2 = st.columns(2)

with col1:
    st.header("Camera Input")
    img_file_buffer = st.camera_input("Click 'Take photo' to capture...")

    # Process image only when a new picture is taken
    if img_file_buffer is not None and img_file_buffer != st.session_state.get('img_file_buffer'):
        st.session_state.img_file_buffer = img_file_buffer # Store the new buffer
        # Read the image
        image = Image.open(img_file_buffer)
        # Convert to NumPy array (RGB)
        img_array_rgb = np.array(image)
        # Convert to BGR for OpenCV
        img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)

        # Predict emotion and get image with face rectangle/label
        detected_emotion, processed_image_bgr = predict_emotion(img_array_bgr.copy()) # Use copy

        st.session_state.detected_emotion = detected_emotion
        # Convert processed image back to RGB for display
        if processed_image_bgr is not None:
            st.session_state.processed_image = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
        else:
             # If prediction fails but image was taken, show original BGR converted to RGB
            st.session_state.processed_image = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)


with col2:
    st.header("Result & Recommendation")
    if st.session_state.processed_image is not None:
        st.image(st.session_state.processed_image, caption="Processed Image", use_column_width=True)

        if st.session_state.detected_emotion:
            st.success(f"Detected Emotion: **{st.session_state.detected_emotion}**")

            # Get the lowercase key for the music map
            emotion_key = LABEL_MAP.get(st.session_state.detected_emotion, 'neutral').lower()
            music_link = EMOTION_MUSIC_MAP.get(emotion_key)

            if music_link:
                st.markdown(f"Feeling **{emotion_key}**? Try this:")
                # Display link - use st.link_button for a nicer look
                try:
                     st.link_button("▶️ Play Recommendation on Spotify", music_link)
                except AttributeError: # Fallback for older Streamlit versions
                     st.markdown(f"▶️ [Play Recommendation on Spotify]({music_link})", unsafe_allow_html=True)
            else:
                 st.info("No specific music recommendation found for this emotion.")

        elif st.session_state.img_file_buffer is not None: # Only show warning if an image was actually taken
             st.warning("No face detected in the captured image. Please try again.")
    else:
        st.info("Take a picture using the camera on the left.")

st.markdown("---")
st.caption("Emotion detection using a Keras model and OpenCV. Music links point to Spotify tracks.")