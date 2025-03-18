import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import imageio
import imageio_ffmpeg
import requests

# Ensure ffmpeg is available
imageio_ffmpeg.get_ffmpeg_version()

# Function to download model if not found
def download_model():
    model_path = "human_action_recognition_model.h5"
    model_url = "https://your-public-url.com/model.h5"  # Replace with actual model URL
    if not os.path.exists(model_path):
        st.write("Downloading model...")
        response = requests.get(model_url, stream=True)
        with open(model_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    return model_path

# Load the trained model
@st.cache_resource
def load_model():
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    return model

# Function to extract frames at 2 frames per second
def extract_frames(video_path, fps=2):
    vid = imageio.get_reader(video_path, format='ffmpeg')
    frame_rate = vid.get_meta_data()['fps']
    interval = int(frame_rate // fps)
    frames = []
    timestamps = []
    
    for i, frame in enumerate(vid):
        if i % interval == 0:
            frames.append(frame)
            timestamps.append(i / frame_rate)
    
    return frames, timestamps

# Function to preprocess frames for model input
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))  # Adjust based on model's input
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)  # Ensure batch dimension (1, 128, 128, 3)
    return frame

# Streamlit UI
st.title("Video Action Recognition")

uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
model = load_model()

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name
    
    st.video(video_path)
    
    st.write("Extracting frames...")
    frames, timestamps = extract_frames(video_path)
    st.write(f"Extracted {len(frames)} frames")
    
    results = []
    class_labels = ['calling', 'clapping', 'cycling', 'dancing', 'listening_to_music', 'eating', 'fighting', 'hugging', 'texting', 'drinking', 'running', 'sitting', 'sleeping', 'laughing', 'using_laptop'] # Updated with actual class names
    
    for frame, timestamp in zip(frames, timestamps):
        preprocessed_frame = preprocess_frame(frame)
        predictions = model.predict(preprocessed_frame)
        predicted_class = class_labels[np.argmax(predictions)]
        results.append(f"At {timestamp:.1f} sec: {predicted_class}")
    
    st.write("Classification Results:")
    for result in results:
        st.write(result)
    
    # Clean up temp file
    os.remove(video_path)
