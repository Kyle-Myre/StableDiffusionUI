import os

__VERSION__ = "1.0.0v"
__AUTHOR__  = "SikroxMemer"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATABASE_URL   = os.path.join(BASE_DIR, 'database', 'session.db')
IMAGES_PATH    = os.path.join(BASE_DIR, 'output')
LORA_PATH      = os.path.join(BASE_DIR, 'loRA')
MODELS_PATH    = os.path.join(BASE_DIR, 'models')

def ensure_directories():
    """
    Create directories if they do not exist.
    """
    for path in [IMAGES_PATH, LORA_PATH, MODELS_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

ensure_directories()

CURRENT_IMAGES = [os.path.join(IMAGES_PATH, image) 
                  for image in os.listdir(IMAGES_PATH) 
                  if os.path.isfile(os.path.join(IMAGES_PATH, image))]

FINAL_IMAGES_LIST = CURRENT_IMAGES.copy()

LORA_LIST  = [lora for lora in os.listdir(LORA_PATH) 
              if os.path.isfile(os.path.join(LORA_PATH, lora))]

MODEL_LIST = (
    'John6666/prefect-pony-xl-v1-sdxl',
    'yodayo-ai/clandestine-xl-1.0'
)

def get_lora(lora: str):
    """
    Returns the absolute path for the given LoRA model.
    """
    lora_path = os.path.join(LORA_PATH, lora)
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA model {lora} not found in {LORA_PATH}")
    return lora_path
