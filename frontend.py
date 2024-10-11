import streamlit as st
from PIL import Image
import numpy as np

# Set page configuration with the updated title
st.set_page_config(
    page_title="Background Removal from Photos",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for an attractive UI
st.markdown(
    """
    <style>
    .main {
        background-color: white;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background-color: #F4F4F4;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .stUpload>button {
        background-color: #28a745;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
    }
    .stUpload>button:hover {
        background-color: #218838;
    }
    .header {
        font-size: 3em;
        text-align: center;
        color: #007bff;
        font-family: 'Roboto', sans-serif;
        font-weight: bold;
    }
    .subheader {
        font-size: 1.5em;
        text-align: center;
        color: #666666;
        font-family: 'Roboto', sans-serif;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #007bff;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header of the app
st.markdown('<div class="header">Background Removal from Photos</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload your image, and let the AI remove the background seamlessly!</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Show original and processed images
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    # Display original image
    st.image(image, caption="Original Image", use_column_width=True)

    # Dummy function to simulate background removal (replace this with actual model logic)
    def remove_background_human(img):
        img_array = np.array(img.convert("RGBA"))
        
        # Placeholder processing logic for human segmentation (replace with actual background removal logic)
        mask = np.random.rand(*img_array.shape[:2]) > 0.5  # Dummy mask for simulation, replace with your model's mask
        img_array[~mask] = [255, 255, 255, 0]  # Make background pixels transparent
        
        return Image.fromarray(img_array)

    # Process the image for background removal (using actual model logic)
    processed_image = remove_background_human(image)

    # Display processed image
    st.image(processed_image, caption="Processed Image", use_column_width=True)

# Footer
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)