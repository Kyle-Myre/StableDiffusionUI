from config.config import IMAGES_PATH
from datetime import datetime
import random
import string
import os

def generate_filename_random_string(length=12):
    letters = string.ascii_uppercase + string.digits
    random_string = ''.join(random.choice(letters) for _ in range(length))
    date = datetime.now().strftime("%Y%m%d")
    final_filename = f"{random_string}-{date}.png"
    filename_path  = os.path.join(IMAGES_PATH , final_filename)
    return filename_path

