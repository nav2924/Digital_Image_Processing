import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import urllib.request
import os

# ---------------------- CONFIG ---------------------- #
st.set_page_config(page_title="Hair Disease Detection", layout="centered")  # ✅ MOVE THIS TO TOP

MODEL_URL = "https://huggingface.co/naveen29012004/cnn_model.h5/resolve/main/cnn_model.h5"
MODEL_PATH = "cnn_model.h5"

CLASS_NAMES = ["Alopecia areata", "Head_Lice", "Psoriasis", "Folliculitis"]
IMG_SIZE = (224, 224)

# ---------------------- LOAD MODEL ---------------------- #
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        with st.spinner("🔄 Downloading model... please wait."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("✅ Model downloaded successfully!")

download_model()
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------- PREPROCESSING ---------------------- #
def preprocess_image(image: Image.Image):
    image = np.array(image.convert("RGB"))
    
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    resized = cv2.resize(enhanced, IMG_SIZE)
    resized = resized / 255.0 
    return np.expand_dims(resized, axis=0)

# ---------------------- UI ---------------------- #
st.title("🧠 Hair Disease Detection")
st.write("Upload a scalp image to detect possible hair-related conditions.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.subheader("🔎 Prediction")
    st.markdown(f"**Condition:** `{CLASS_NAMES[predicted_class]}`")
    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
