import streamlit as st
from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Set page configuration for dark theme
st.set_page_config(page_title="Text to Image Generator", layout="centered")

# Set dark theme for Streamlit UI
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stButton>button {
        background-color: #6200ea;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 12px;
    }
    .stTextInput>div>div>input {
        background-color: #333333;
        color: white;
        border: 1px solid #6200ea;
    }
    .stTextInput>div>label {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# Load model
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda")

# Streamlit UI
st.title("Text to Image Generator")

# Input: Prompt for image generation
prompt = st.text_input("Enter your prompt", "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")

# Image Style Selection
image_style = st.selectbox(
    "Select image style",
    ["Detailed", "Blur", "Fantasy", "Realistic", "Cartoon", "Surreal"]
)

# Theme Selection
theme = st.radio("Select theme", ["Dark", "Vibrant", "Muted Colors", "Nature", "Cosmic"])

# Resolution Slider
resolution = st.slider("Select resolution", 256, 1024, 512, 64)

# Guidance Scale Slider
guidance_scale = st.slider("Select guidance scale", 7.0, 15.0, 9.0, 0.1)

# Generate Image Button
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Modify prompt based on user selections
        prompt_with_style = f"{prompt}, {image_style.lower()} style, {theme.lower()} theme"
        
        # Generate image based on the modified prompt and selected settings
        image = pipe(prompt_with_style, num_inference_steps=resolution // 64, guidance_scale=guidance_scale).images[0]
        
        # Display the generated image
        st.image(image, caption="Generated Image", use_column_width=True)
        st.success("Image generated successfully!")

# Optionally, save the generated image
if st.button("Save Image"):
    image.save("generated_image.png")
    st.success("Image saved successfully!")