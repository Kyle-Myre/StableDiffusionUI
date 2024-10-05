import streamlit as st
import sys
import json
sys.dont_write_bytecode=True
from functions import Generator
from functions import Pipeline

@st.cache_data
def diffuser(repo_url:str):
    return Pipeline(repo_url=repo_url)

st.set_page_config(page_title='Stable Diffusion UI' , page_icon='./public/favicon.ico' , layout='wide')


class App:
    """
    - Stable Diffusion UI App main class.
    """

    result:str

    class Container:
        form  , result = st.columns(2)

    def __init__(self) -> None:

        self.result: str = ""

        with self.Container.form:

            self.set_title("Stable Diffusion UI")
            self.set_description()
            self.set_form()

        with self.Container.result:
            self.set_result()
        
    def set_description(self) -> None:
        file = open('./docs/description.html')
        doc:str  = file.read()
        file.close()
        st.markdown(doc ,unsafe_allow_html=True)
        st.empty()

    def set_title(self , text:str) -> st._DeltaGenerator:
        st.subheader(text)

    def set_form(self) -> None:
        self.status = st.empty()

        with st.form('main_form'):

            with open('config.json' , 'r') as config:
                self.config:dict = json.load(config)

            self.model            = st.selectbox(label='Model' , options=self.config['models'])

            self.lora             = st.multiselect('LoRA', ['1' , '2' , '3'])
            self.positive_prompt  = st.text_area('Prompt', placeholder='1boy, astronaut, green trees, etc, ...')
            self.negative_prompt  = st.text_area('Negative', placeholder='explicit, bad quality, nsfw, etc, ...')
            self.steps_prompt     = st.slider('Steps', min_value=1, value=30 , max_value=100)
            self.gradience_prompt = st.slider('Gradience', min_value=1, value=8 ,  max_value=100)
            self.height_prompt    = st.number_input('Height' , value=1352.00)
            self.width_prompt     = st.number_input('Width' , value=784.00)

            self.submit_button    = st.form_submit_button('Submit')

            try:
                pipeline = diffuser(self.model)
            except Exception as error:
                print(f"[red]{error}[/]")

            if self.submit_button:
                pipeline.pipeline()

                self.result = Generator().generate(
                    pipeline , 
                    self.positive_prompt , 
                    self.negative_prompt , 
                    (self.width_prompt , self.height_prompt),
                    self.gradience_prompt,
                    self.steps_prompt
                )

    def set_result(self) -> None:
        st.subheader("")
        try:
            st.image(self.result , width=600)
        except Exception:
            st.empty()



app = App()