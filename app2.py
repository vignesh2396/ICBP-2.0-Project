import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import io
import base64

# Load ONNX model
@st.cache_resource
def load_model():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession('AnimeGANv3_Hayao_36.onnx', providers=providers)
    return session

session = load_model()

# Preprocess image
def process_image(img, x8=True):
    h, w = img.shape[:2]
    if x8:
        def to_8s(x):
            return 256 if x < 256 else x - x % 8
        img = cv2.resize(img, (to_8s(w), to_8s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return img

# Run inference
def convert_image(img, scale):
    input_name = session.get_inputs()[0].name
    img_input = np.expand_dims(process_image(img), axis=0)
    output = session.run(None, {input_name: img_input})[0]
    image_out = (np.squeeze(output) + 1.0) / 2 * 255
    image_out = np.clip(image_out, 0, 255).astype(np.uint8)
    image_out = cv2.resize(image_out, (scale[0], scale[1]))
    return image_out

# Download link helper
def get_image_download_link(img, filename='converted.png'):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}"><button style="padding: 10px 20px;">Download Image</button></a>'

# App layout
st.title("AnimeGANv3 - Image Stylizer")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

scale_option = st.selectbox("Choose Output Scale", ["Original", "2x", "0.5x"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, 1)

    h, w = original.shape[:2]

    if scale_option == "2x":
        new_w, new_h = int(w * 2), int(h * 2)
    elif scale_option == "0.5x":
        new_w, new_h = int(w * 0.5), int(h * 0.5)
    else:
        new_w, new_h = w, h

    # Run model
    output_image = convert_image(original, (new_h, new_w))

    # Convert to PIL for download
    output_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    # Show results in columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(original, channels="BGR")
        st.caption(f"Width: {w} px | Height: {h} px")

    with col2:
        st.subheader("Stylized")
        st.image(output_image, channels="RGB")
        st.caption(f"Width: {new_w} px | Height: {new_h} px")
        st.markdown(get_image_download_link(output_pil), unsafe_allow_html=True)
