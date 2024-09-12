from dotenv import load_dotenv
from threading import Thread
from pyngrok import ngrok
import os

load_dotenv()

ngrok.set_auth_token(os.getenv('NGROK'))

def run(): os.system("streamlit run app/web.py --server.port 6060")

thread = Thread(target=run)
thread.start()