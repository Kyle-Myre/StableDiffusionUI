from datetime import datetime
from paths import IMAGES_PATH
import safetensors.torch
import random , string
import torch , os

def generate_filename_random_string(length=12):
    letters = string.ascii_uppercase + string.digits
    random_string = ''.join(random.choice(letters) for _ in range(length))
    date = datetime.now().strftime("%Y%m%d")
    final_filename = f"{random_string}-{date}.png"
    filename_path  = os.path.join(IMAGES_PATH , final_filename)
    return filename_path

def reshape_lora_weights(lora_weights, expected_shape):
    reshaped_weights = {}
    for name, weight in lora_weights.items():
        if weight.shape != expected_shape and len(weight.shape) > 2:
            reshaped_weights[name] = weight.view(expected_shape)
        else:
            reshaped_weights[name] = weight
    return reshaped_weights

def load_lora_weights_with_reshape(model, lora_path):
    lora_weights = safetensors.torch.load_file(lora_path)
    expected_shape = torch.Size([32, 640])
    lora_weights = reshape_lora_weights(lora_weights, expected_shape)
    for name, param in model.named_parameters():
        if name in lora_weights:
            param.data += lora_weights[name].to(param.device)