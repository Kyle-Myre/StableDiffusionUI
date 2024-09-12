import uuid
import os
import gc
import torch
from transformers import BertTokenizer
from safetensors.torch import load_file as load_safetensors
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# Assuming IMAGES_PATH is imported from config
from config.config import IMAGES_PATH


def generate_filename_random_string():
    """Generates a unique filename for the image."""
    return os.path.join(IMAGES_PATH, f"generated_image_{uuid.uuid4()}.png")


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

    return tokenizer.convert_tokens_to_string(tokens)


def generate(pipe, image_component, prompt: str, negative: str, width: int, height: int, guidance_scale: float, steps: int):
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
    filename = generate_filename_random_string()
    processed_prompt = prepare_prompt(prompt, max_length=77)

    with torch.no_grad():
        # Ensure the image is generated on GPU without gradients to optimize memory
        image = pipe(
            processed_prompt,
            negative_prompt=negative,
            width=int(width),
            height=int(height),
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        ).images[0]

        image.save(filename)

    image_component.image(filename)

    # Explicit cleanup for GPU memory
    del image
    torch.cuda.empty_cache()
    gc.collect()


def set_up_pipeline(repo_url, clip_skip=2, device="cuda"):
    """
    Configure Stable Diffusion XL pipeline and download a model from Hugging Face if one doesn't exist.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    pipe = StableDiffusionXLPipeline.from_pretrained(
        repo_url,
        torch_dtype=torch.float16,
        use_safetensors=True,
        custom_pipeline="lpw_stable_diffusion_xl",
        add_watermarker=False,
        device_map="balanced" if device == "cuda" else None
    )

    # Memory-efficient scheduler and attention
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    # Clip skip to control overfitting to the prompt
    if hasattr(pipe, 'config'):
        pipe.config.clip_skip = clip_skip

    return pipe


def load_lora_weights(pipe, lora_paths, alphas=None):
    """
    Load and apply multiple LoRA weights sequentially to the pipeline.

    Args:
        pipe: The Stable Diffusion pipeline.
        lora_paths: A list of paths to LoRA models (.safetensors files or repos).
        alphas: Optional list of scaling factors for blending LoRAs. If None, equal scaling is applied.
    """
    if alphas is None:
        alphas = [1.0] * len(lora_paths)

    unet_state_dict = pipe.unet.state_dict()

    for i, lora_path in enumerate(lora_paths):
        # Load weights depending on file type
        lora_weights = load_safetensors(lora_path) if lora_path.endswith('.safetensors') else torch.load(lora_path)

        # Apply LoRA weights with alpha scaling
        for key, value in lora_weights.items():
            if key in unet_state_dict:
                unet_state_dict[key].copy_(unet_state_dict[key] * (1 - alphas[i]) + value * alphas[i])

        print(f"LoRA weights from {lora_path} loaded with alpha {alphas[i]}.")

    return pipe
