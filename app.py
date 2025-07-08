import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Preprocess function for images
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

# Prediction for a single frame
def predict_frame(frame):
    processed = preprocess_image(frame)
    pred = model.predict(processed)[0][0]
    label = "FAKE" if pred > 0.5 else "REAL"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence * 100

# Aggregate predictions for video
def analyze_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    stframe = st.empty()
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            label, confidence = predict_frame(frame)
            results.append((label, confidence))

            # Overlay prediction on frame
            text = f"{label} ({confidence:.1f}%)"
            color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            stframe.image(frame, channels="BGR", caption=f"Frame {frame_idx}")

    cap.release()

    # Aggregate final result
    fake_count = sum(1 for r in results if r[0] == "FAKE")
    real_count = sum(1 for r in results if r[0] == "REAL")
    final_label = "FAKE" if fake_count > real_count else "REAL"
    confidence_score = (fake_count / len(results)) * 100 if final_label == "FAKE" else (real_count / len(results)) * 100

    return final_label, confidence_score

# Streamlit UI
st.title("🎥 Deepfake Video Detector")
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4", "avi"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        img_array = np.array(img)
        label, confidence = predict_frame(img_array)
        st.success(f"Prediction: **{label}** ({confidence:.2f}% confidence)")

    elif uploaded_file.type.startswith('video'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.info("Analyzing video... This may take a while for longer videos.")
        final_label, confidence_score = analyze_video(video_path, frame_skip=15)

        st.success(f"Final Prediction: **{final_label}** ({confidence_score:.2f}% confidence)")
