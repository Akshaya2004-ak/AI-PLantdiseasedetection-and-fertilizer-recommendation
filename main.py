import streamlit as st
import numpy as np
import pickle
import cv2
import google.generativeai as genai
from keras.preprocessing.image import img_to_array

# -----------------------------------
# Configuration
# -----------------------------------
st.set_page_config(
    page_title="AI-Powered Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# Load the trained CNN model and label binarizer
model = pickle.load(open("cnn_model.pkl", "rb"))
label_binarizer = pickle.load(open("label_transform.pkl", "rb"))

# Configure Gemini API key
genai.configure(api_key="AIzaSyAIsluRyWX6jocF2d7hNVkO66GIqJ-5-5Q")

# -----------------------------------
# Image Preprocessing Function
# -----------------------------------
def preprocess_image(image_data):
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (256, 256))
    img = img_to_array(img)
    img = img.astype("float") / 225.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------------
# UI Layout
# -----------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57; font-family: Helvetica, sans-serif;'>
        🌿 AI-Powered Plant Disease Detector
    </h1>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a clear leaf image", type=["jpg", "jpeg", "png"])

# -----------------------------------
# Prediction Logic
# -----------------------------------
if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with st.spinner("Analyzing the image..."):
        image = preprocess_image(uploaded_file)
        prediction = model.predict(image)
        predicted_disease = label_binarizer.classes_[np.argmax(prediction)]

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("#### Detected Disease")
        st.markdown(
            f"<div style='background-color:#e0f7e9; padding: 12px; "
            f"border-left: 5px solid #2E8B57; border-radius: 5px;'>"
            f"<strong>{predicted_disease}</strong></div>",
            unsafe_allow_html=True
        )

    with st.spinner("Generating treatment suggestions..."):
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"The detected plant disease is {predicted_disease}. "
            "Please provide only the required fertilizer and pesticide treatment."
        )
        response = model_gemini.generate_content(prompt)
        suggestions = response.text

    st.markdown("#### 🌱 Suggested Remedies and Recovery Plan")
    st.markdown(suggestions)

    # Divider
    st.markdown("<hr style='border: none; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)
