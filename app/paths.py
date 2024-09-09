import os

DATABASE_URL   =  os.path.abspath(os.path.join(os.path.dirname(__file__) , 'database' , 'session.db'))
IMAGES_PATH    =  os.path.abspath(os.path.join(os.path.dirname(__file__) , 'output'))
LORA_PATH      =  os.path.abspath(os.path.join(os.path.dirname(__file__) , 'lora'))

CURRENT_IMAGES =  [image for image in os.listdir(IMAGES_PATH)]
FINALL_IMAGES_LIST = []

for image in CURRENT_IMAGES:
    image = os.path.join(os.path.dirname(__file__) , 'output' , image)
    FINALL_IMAGES_LIST.append(image)


LORA_LIST = [lora for lora in os.listdir(LORA_PATH)]

def get_lora(lora:str):
    return os.path.join(LORA_PATH , lora)

print(IMAGES_PATH)