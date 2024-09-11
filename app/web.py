# from main import generate, configure_pipeline
from paths   import FINALL_IMAGES_LIST , LORA_LIST
from pipeline import configure_pipeline
from session import latest_records
from worker import on_submit

from config import __VERSION__
import streamlit as st
import pandas as pd

st.set_page_config(layout='wide')
form , output = st.columns(2)

@st.cache_resource
def cache_pipeline(model_option:str):
    return configure_pipeline(model_option)[1]



with form:
    st.title(f'Stable Diffusion {__VERSION__}')
    st.markdown(
        """
        <div style='text-align:justify;'>This App provides a Interface for running Stable Diffusion, 
        a state-of-the-art text-to-image generation model. It integrates Hugging Face's 
        `diffusers` library along with various other dependencies to streamline the 
        generation process. Additionally, the App supports database operations 
        and visualization, making it versatile for various use cases.</div>
        <br/>
        """ , unsafe_allow_html=True
    )

    task_status = st.empty()

    model , lora = st.tabs(["Models" , "LoRA's"])

    with model:
        status       = st.empty()
        model_option = st.selectbox("Select Available Model",('John6666/prefect-pony-xl-v1-sdxl','yodayo-ai/clandestine-xl-1.0'))
        
    with lora:
        lora_combo = st.multiselect("Select Available Model Available LoRA" , LORA_LIST)

    if model_option:
        with status.status("Preparing Model" , expanded=False):
            pipe = cache_pipeline(model_option)
            st.empty()

    generating_status = st.empty()

    with st.form("main_form"):
        positive_prompt  = st.text_area('Prompt', placeholder='1boy, astronaut, green trees, etc, ...')
        negative_prompt  = st.text_area('Negative', placeholder='explicit, bad quality, nsfw, etc, ...')
        steps_prompt     = st.slider('Steps', min_value=1, value=30 , max_value=100)
        gradience_prompt = st.slider('Gradience', min_value=1, value=8 ,  max_value=100)
        height_prompt    = st.number_input('Height' , value=1352.00)
        width_prompt     = st.number_input('Width' , value=784.00)
        submit_button    = st.form_submit_button('Submit')

        

with output:
    st.subheader('Output')
    image = st.empty()
    image.image(FINALL_IMAGES_LIST[0] , caption=FINALL_IMAGES_LIST[-1])


if submit_button:
    with generating_status.status("Generating Image"):
        try:
            on_submit(pipe=pipe,lora_options=lora_combo,image_component=image , 
                height=height_prompt,width=width_prompt,guidance_scale=gradience_prompt,
                negative=negative_prompt,prompt=positive_prompt,steps=steps_prompt,
            )
            task_status.success('Successfully Generated Task')
        except Exception as error:
            task_status.error(error)