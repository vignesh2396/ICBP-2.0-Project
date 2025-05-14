import streamlit as st
from PIL import Image
import torch
from io import BytesIO
import base64

# Load the model once and cache it
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device).eval()
    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)
    return face2paint, model

face2paint, model = load_model()

# App UI
st.title("🎨 AnimeGAN2 Face Stylizer")
st.markdown("Upload an image, choose output scale, and download the anime-styled version.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Scaling options
scale_factor = st.selectbox("Select output size scale", options=["Original", "1.5x", "2x", "3x"])
scale_map = {"Original": 1, "1.5x": 1.5, "2x": 2, "3x": 3}

# Process if image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    orig_width, orig_height = image.size

    with st.spinner("Generating anime version..."):
        output = face2paint(model, image, side_by_side=False)

        # Scale output
        scale = scale_map[scale_factor]
        if scale != 1:
            new_width = int(output.size[0] * scale)
            new_height = int(output.size[1] * scale)
            output = output.resize((new_width, new_height))
        else:
            new_width, new_height = output.size

        # Side-by-side layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            # st.image(image, use_container_width=True)
            st.image(image)
            st.caption(f"Size: {orig_width} x {orig_height}")

        with col2:
            st.markdown("### Output")
            # st.image(output, use_container_width=True)
            st.image(output)
            st.caption(f"Size: {new_width} x {new_height}")

        # Download button using form (for a real button look)
        buf = BytesIO()
        output.save(buf, format="PNG")
        byte_im = buf.getvalue()
        b64 = base64.b64encode(byte_im).decode()
        download_button_html = f"""
            <a href="data:image/png;base64,{b64}" download="anime_output.png">
                <button style="
                    background-color: #4CAF50;
                    color: white;
                    padding: 0.5em 1em;
                    border: none;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    margin-top: 1em;
                ">
                    📥 Download Output Image
                </button>
            </a>
        """
        st.markdown(download_button_html, unsafe_allow_html=True)