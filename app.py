import os
from io import BytesIO
from tempfile import TemporaryDirectory

import boto3
import numpy as np
import streamlit as st
import toml
from keras.models import load_model
from PIL import Image
from streamlit_image_comparison import image_comparison
from tensorflow import keras


@st.cache_resource
def load_cached_model():
    with open("./.streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
        s3_params = secrets["s3"]

    s3 = boto3.client(
        "s3",
        aws_access_key_id=s3_params["key"],
        aws_secret_access_key=s3_params["secret"],
        region_name="us-east-1",
    )

    with TemporaryDirectory() as tmpdir:
        local_model_path = os.path.join(tmpdir, s3_params["model_file"])
        s3.download_file(
            s3_params["bucket_name"], s3_params["model_file"], local_model_path
        )

        model = load_model(local_model_path, compile=False)

    return model


st.set_page_config(
    page_title="Text Enhancer",
    page_icon=":memo:",
    layout="wide",
)

model = load_cached_model()

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

st.header(":memo: :orange[Text & Formula Image Enhancer]", anchor=False)

st.text(
    """
Utilize our free upscaling tool to enhance the clarity of your text and formula screenshots. 
Say goodbye to blurry images in PowerPoint or Word. 
Our upscaler improves the quality of your pictures, ensuring sharper and clearer visuals for your presentations and documents.
"""
)

st.divider()

example_img_col, upload_col, settings_col = st.columns([1, 1, 1])

with example_img_col:
    image_comparison(
        img1="./img/image_before_enhancing.png",
        img2="./img/image_after_enhancing.png",
        label1="Before",
        label2="After",
        width=324,
        starting_position=50,
    )

with upload_col:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

with settings_col:
    increase_image_size = (
        st.radio(
            ":abacus: **Enhancement Mode**",
            options=["Increase Size", "Enhance Quality"],
            index=0,
        )
        == "Increase Size"
    )
    process_image_clicked = st.button(":rocket: Enhance Uploaded Image")

st.divider()

if uploaded_file is None and process_image_clicked:
    st.error("Please upload an image first.")
    st.divider()


if uploaded_file is not None:
    input_image_col, increase_sized_image_col = st.columns(2)

    image = Image.open(uploaded_file)
    with input_image_col:
        st.write(f"Uploaded Image ({image.size[0]}x{image.size[1]})")
        st.image(image, caption="Uploaded Image")

    if process_image_clicked:
        enhanced_image = enhance_image(
            image.copy(), model, increase_size=increase_image_size
        )

        with increase_sized_image_col:
            st.write(
                f"Enhanced Image ({enhanced_image.size[0]}x{enhanced_image.size[1]})"
            )
            st.image(enhanced_image, caption="Enhanced Image")

            # Image Download
            buf = BytesIO()
            enhanced_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label=":rocket: Download Enhanced Image",
                data=byte_im,
                file_name="enhanced_text_image.png",
            )

    st.divider()


with st.columns(2)[0]:
    with st.expander(":thinking_face: How it works"):
        st.write(
            """
        Traditional image upscaling methods use interpolation methods to increase the size of an image. 
        Those methods are not optimized for text and formula images, which are often blurry and difficult to read. 
        We have trained a Deep Learning model to enhance the clarity of text and formula images, ensuring sharper and clearer visuals for your presentations and documents.
        """
        )

        st.write(
            """
            Feel free to propose new features or report bugs [here](https://github.com/jkrol21/text_enhancer).
        """
        )
