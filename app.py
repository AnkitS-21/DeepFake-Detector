import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import tempfile

# 📦 Load ResNeXt model
MODEL_PATH = "resnext50_deepfake.pth"

# Build the same architecture
model = models.resnext50_32x4d(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Binary classifier (output logits)

# Load saved weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Preprocessing pipeline (matches training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 🖼️ Preprocess image
def preprocess_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img

# 🔥 Prediction for a single frame
def predict_frame(frame):
    input_tensor = preprocess_image(frame)
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
    label = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
    return label, confidence

# 🎥 Analyze video frame-by-frame
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

            # Overlay prediction
            text = f"{label} ({confidence:.1f}%)"
            color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            stframe.image(frame, channels="BGR", caption=f"Frame {frame_idx}")

    cap.release()

    # Final decision
    fake_count = sum(1 for r in results if r[0] == "FAKE")
    real_count = sum(1 for r in results if r[0] == "REAL")
    final_label = "FAKE" if fake_count > real_count else "REAL"
    confidence_score = (fake_count / len(results)) * 100 if final_label == "FAKE" else (real_count / len(results)) * 100

    return final_label, confidence_score

# 🎛 Streamlit UI
st.title("🎥 Deepfake Detector (PyTorch - ResNeXt50)")
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
