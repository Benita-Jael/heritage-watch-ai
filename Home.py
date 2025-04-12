import streamlit as st
import base64

# Page config
st.set_page_config(page_title="Heritage Watch AI", layout="wide")

# Set background image from local file
def set_bg_image(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .custom-button {{
            display: flex;
            justify-content: center;
            margin-top: 60vh;
        }}
        .button-text {{
            font-size: 22px;
            font-weight: bold;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# Set the uploaded background image
set_bg_image("assets/background.jpg")

# Centered "Let's Detect" button
st.markdown("<div class='custom-button'>", unsafe_allow_html=True)
if st.button("Let's Detect"):
    st.switch_page("pages/Main.py")
st.markdown("</div>", unsafe_allow_html=True)
