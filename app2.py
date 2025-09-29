import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import os

# -------------------------------
# Load models and labels
# -------------------------------
seq_model = load_model("gesture_model_seq.h5")
img_model = load_model("gesture_model_img.h5")
label_dict = np.load("gesture_labels.npy", allow_pickle=True).item()
reverse_labels = {v:k for k,v in label_dict.items()}

IMG_SIZE = (64, 64)
FRAMES_PER_CLIP = 20

st.title("🤚 Hand Gesture Recognition App")
st.write("Upload an **Image**, **PDF**, or **Video** to predict gestures.")

# -------------------------------
# Helper functions
# -------------------------------
def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_image(img):
    img_array = preprocess_image(img)
    pred = img_model.predict(img_array)
    return reverse_labels[np.argmax(pred)]

def predict_pdf(pdf_file):
    pages = convert_from_path(pdf_file)
    results = []
    for page in pages:
        img = np.array(page)
        results.append(predict_image(img))
    return results

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    result = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = frame / 255.0
        frames.append(frame)
        if len(frames) == FRAMES_PER_CLIP:
            X = np.expand_dims(np.array(frames), axis=0)
            pred = seq_model.predict(X)
            result = reverse_labels[np.argmax(pred)]
            frames = []
    cap.release()
    return result

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("Choose an Image, PDF, or Video", type=["jpg", "png", "pdf", "mp4"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    if file_ext in ["jpg", "png"]:
        img = cv2.imread(temp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        prediction = predict_image(img)
        st.success(f"🖐 Predicted Gesture: **{prediction}**")

    elif file_ext == "pdf":
        st.info("Processing PDF pages...")
        predictions = predict_pdf(temp_path)
        for i, pred in enumerate(predictions, 1):
            st.write(f"Page {i}: {pred}")

    elif file_ext == "mp4":
        st.info("Processing Video...")
        prediction = predict_video(temp_path)
        st.video(temp_path)
        if prediction:
            st.success(f"🖐 Predicted Gesture from Video: **{prediction}**")
        else:
            st.warning("Could not detect a complete gesture in the video.")

    # Delete temporary file
    os.remove(temp_path)
