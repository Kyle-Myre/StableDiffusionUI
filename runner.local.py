from threading import Thread
from pyngrok import ngrok
import os

ngrok.set_auth_token("")

def run():
    os.system("streamlit run app/web.py --server.port 5439")


thread = Thread(target=run)
thread.start()


public_url = ngrok.connect(addr='5439' , proto='http' , bind_tls=True)
print(f'Running at {public_url}')