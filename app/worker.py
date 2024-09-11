from utilities import generate_filename_random_string
from utilities import load_lora_weights_with_reshape

from session import insert_record
from paths import get_lora

def on_submit(config , lora_options: list[str],image_component,prompt: str,negative: str,width: float,height: float,guidance_scale: int,steps: int):

    if lora_options:
        lora_weights = [get_lora(lora) for lora in lora_options]  # Gather LoRA weights
        for lora_weight in lora_weights:
            load_lora_weights_with_reshape(config[1].unet, lora_weight)

    tokens = config[0](prompt, truncation=True, max_length=77)
    prompt_truncated = config[0].decode(tokens['input_ids'], skip_special_tokens=True)

    filename = generate_filename_random_string()
    image = config[1](
        prompt_truncated,
        negative_prompt=negative,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    ).images[0]
    image.save(filename)
    image_component.image(filename)
    insert_record(prompt, negative, steps, guidance_scale)