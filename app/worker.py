from utilities import load_lora_weights_with_reshape , reshape_lora_weights
from utilities import generate_filename_random_string
from session import insert_record
from paths import get_lora


def on_submit(pipe , lora_options:list[str] , image_component , number:int , prompt:str , negative:str , width:float , height:float , guidance_scale:int , steps:int):

    for lora in lora_options:
        load_lora_weights_with_reshape(pipe.unet, get_lora(lora))

    for _ in range(number):
        filename = generate_filename_random_string()

        image = pipe(prompt , negative_prompt=negative,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        ).images[0]

        image.save(filename)
        insert_record(prompt , negative , steps , guidance_scale)

        image_component.image('')