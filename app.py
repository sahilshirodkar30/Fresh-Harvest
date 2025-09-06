import streamlit as st
from model_help import predict
from PIL import Image
import io

st.set_page_config(page_title="Fresh Harvest Detection", layout="centered")

st.title("Fresh Harvest Detection")
st.write("Upload or drag & drop an image, and the model will classify the type of Fresh Harvested .")

# File uploader with drag & drop
uploaded_image = st.file_uploader("Drag and drop an image here or click to upload", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Open and display uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temporarily for model input
    image_path = "temp_uploaded.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Predict
    with st.spinner("Analyzing image..."):
        prediction = predict(image_path)

    st.success(f"✅ Predicted : **{prediction}**")
