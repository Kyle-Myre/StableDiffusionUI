try:
    import streamlit as st
    import yaml
except (ImportError , ImportWarning) as error:
    print(error)
    exit(-1)

st.set_page_config(page_title='Stable Diffusion UI' , page_icon='./public/favicon.ico' , layout='wide')
class App(object):
    """
    - Stable Diffusion UI App main class.
    """
    class Container:
        form  , result = st.columns(2)

    def set_models(self) -> None:
        with open('config.yaml' , 'r') as config:
            self.config:dict = yaml.safe_load(config)
        st.selectbox(label='Model' , options=self.config['models'])

    def __init__(self) -> None:
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
            self.set_models()
            st.multiselect('LoRA', ['1' , '2' , '3'])
            self.positive_prompt  = st.text_area('Prompt', placeholder='1boy, astronaut, green trees, etc, ...')
            self.negative_prompt  = st.text_area('Negative', placeholder='explicit, bad quality, nsfw, etc, ...')
            self.steps_prompt     = st.slider('Steps', min_value=1, value=30 , max_value=100)
            self.gradience_prompt = st.slider('Gradience', min_value=1, value=8 ,  max_value=100)
            self.height_prompt    = st.number_input('Height' , value=1352.00)
            self.width_prompt     = st.number_input('Width' , value=784.00)
            self.submit_button    = st.form_submit_button('Submit')


    def set_result(self) -> None:
        st.subheader("")
        st.image('./output/f7386c16-c881-4570-bfc9-a54f945c13c6.jpeg' , width=600)



app = App()