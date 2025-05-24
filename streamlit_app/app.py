import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import urllib.request
import os
from disease_info import disease_data, fallback_message

st.set_page_config(page_title="Hair Disease Detection", layout="centered")

# Load model from Hugging Face once and cache it
@st.cache_resource
def load_model():
    model_path = "cnn_model.h5"
    if not os.path.exists(model_path):
        url = "https://huggingface.co/naveen29012004/cnn_model.h5/resolve/main/cnn_model.h5"
        urllib.request.urlretrieve(url, model_path)
    return tf.keras.models.load_model(model_path)

model = load_model()

# Class names
class_names = ["Alopecia areata", "Head_Lice", "Psoriasis", "Folliculitis"]

# Constants
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7

# Image preprocessing function
def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    resized = cv2.resize(enhanced, IMG_SIZE)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Streamlit UI
st.title("🩺 Hair Disease Detector")
st.write("Upload a scalp image to detect common hair/scalp conditions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        input_image = preprocess_image(image)
        predictions = model.predict(input_image)
        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        if confidence >= CONFIDENCE_THRESHOLD:
            disease = class_names[predicted_index]
            info = disease_data.get(disease, fallback_message)
        else:
            disease = "Uncertain Prediction"
            info = fallback_message

        st.markdown(f"## 🧠 Diagnosis: **{disease}**")
        st.markdown(f"**🔬 Confidence:** `{confidence:.2f}`")
        st.markdown(f"### 📄 About the condition:")
        st.write(info["description"])
        st.markdown("### 💡 Suggested Remedies:")
        for remedy in info["remedies"]:
            st.markdown(f"- {remedy}")
