# from main import generate, configure_pipeline
from paths   import FINALL_IMAGES_LIST , LORA_LIST

from session import latest_records
from config import __VERSION__

import streamlit as st
import pandas as pd

@st.cache_resource
def generate_pipe():
    ...

st.set_page_config(layout='wide')
form , output = st.columns(2)

with form:
    st.title(f'Stable Diffusion {__VERSION__}')
    st.markdown(
        """
        <div style='text-align:justify;'>This App provides a Interface for running Stable Diffusion, 
        a state-of-the-art text-to-image generation model. It integrates Hugging Face's 
        `diffusers` library along with various other dependencies to streamline the 
        generation process. Additionally, the App supports database operations 
        and visualization, making it versatile for various use cases.</div>
        """ , unsafe_allow_html=True
    )
    model , lora = st.tabs(["Models" , "LoRA's"])

    with model:
        model_option = st.selectbox("Select Available Model",('John6666/prefect-pony-xl-v1-sdxl','yodayo-ai/clandestine-xl-1.0'),)
    with lora:
        lora_combo = st.multiselect("Select Available Model Available LoRA" , LORA_LIST)

    positive_prompt = st.text_area('Prompt', placeholder='1boy, astronaut, green trees, etc, ...')
    negative_prompt = st.text_area('Negative', placeholder='explicit, bad quality, nsfw, etc, ...')
    steps_prompt = st.slider('Steps', min_value=1, value=30 , max_value=100)
    gradience_prompt = st.slider('Gradience', min_value=1, value=8 ,  max_value=100)
    height_prompt = st.number_input('Height' , value=1352.00)
    width_prompt = st.number_input('Width' , value=784.00)
    submit_button = st.button('Submit')
    history , options = st.tabs(["History", "About"])

    with history:
        data = latest_records()
        df = pd.DataFrame(data[1:], columns=data[0])
        st.table(df)

    with options:
        st.markdown("""
            ### Stable Diffusion 1.0.0v By Sikrox Memer
            <https://github.com/SikroxMemer>
            Made Under MIT License Feel Free to use it.
        """)

with output:
    st.subheader('Output')
    try:
        image = st.image(FINALL_IMAGES_LIST[1], caption="cat.png")
    except:
        st.write(FINALL_IMAGES_LIST)