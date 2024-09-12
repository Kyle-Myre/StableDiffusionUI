from transformers  import BertTokenizer
from config.config import IMAGES_PATH
import torch

def generate_filename_random_string():
    import uuid , os
    filename = f"generated_image_{uuid.uuid4()}.png"
    return os.path.join(IMAGES_PATH , filename)

from transformers import BertTokenizer

def prepare_prompt(prompt: str, max_length: int = 77):
    """
    Prepares the prompt by truncating it to the maximum token length allowed by the model.
    
    Args:
        prompt: The input text prompt.
        max_length: The maximum token length (77 for most CLIP models).
    
    Returns:
        truncated_prompt: The truncated prompt ready for tokenization.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(prompt)

    if len(tokens) > max_length:
        print(f"Prompt too long. Truncating from {len(tokens)} tokens to {max_length} tokens.")
        tokens = tokens[:max_length]

    truncated_prompt = tokenizer.convert_tokens_to_string(tokens)
    return truncated_prompt


def generate(
        pipe,
        image_component,
        prompt: str,
        negative: str,
        width: int,
        height: int,
        guidance_scale: float,
        steps: int
    ):
    """
    Generate an image with optimized performance.
    
    Args:
        pipe: The Stable Diffusion pipeline.
        image_component: The component to display the generated image.
        prompt: The text prompt for image generation.
        negative: Negative prompt to exclude unwanted details.
        width: Width of the generated image (must be int).
        height: Height of the generated image (must be int).
        guidance_scale: Control over adherence to the prompt (higher values = more adherence).
        steps: Number of inference steps.
    """
    width = int(width)
    height = int(height)
    filename = generate_filename_random_string()
    processed_prompt = prepare_prompt(prompt , max_length=77)

    with torch.no_grad():
        image = pipe(
            processed_prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        ).images[0]
        image.save(filename)
    image_component.image(filename)
    del image
    torch.cuda.empty_cache()