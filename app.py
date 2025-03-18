import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import ffmpeg

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("human_action_recognition_model.h5")  # Update with the correct path
    return model

# Function to extract frames at 2 frames per second
def extract_frames(video_path, fps=2):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    frame_rate = eval(video_stream['r_frame_rate'])
    interval = int(frame_rate // fps)
    
    out, _ = (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=fps)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, capture_stderr=True)
    )
    
    frame_size = (int(video_stream['width']), int(video_stream['height']))
    frame_count = len(out) // (frame_size[0] * frame_size[1] * 3)
    frames = [
        np.frombuffer(out[i * frame_size[0] * frame_size[1] * 3:(i + 1) * frame_size[0] * frame_size[1] * 3], np.uint8)
        .reshape((frame_size[1], frame_size[0], 3))
        for i in range(frame_count)
    ]
    timestamps = [i / fps for i in range(frame_count)]
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
    class_labels = ['calling', 'clapping', 'cycling', 'dancing', 'listening_to_music', 'eating', 'fighting', 'hugging', 'texting', 'drinking', 'running', 'sitting', 'sleeping', 'laughing', 'using_laptop']  # Update with actual class names
    
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
