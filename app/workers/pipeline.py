import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from safetensors.torch import load_file as load_safetensors


def set_up_pipeline(repo_url, clip_skip=2, device="cuda"):
    """
    Configure Stable Diffusion XL pipeline and download a model from Hugging Face if one doesn't exist.
    Switches between CPU and GPU.
    """
    if torch.cuda.is_available() and device == "cuda":
        print("Using GPU")
        torch_device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    else:
        print("Using CPU")
        torch_device = torch.device("cpu")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        repo_url,
        torch_dtype=torch.float16 if torch_device.type == "cuda" else torch.float32,  # Use mixed precision for GPU
        use_safetensors=True,
        custom_pipeline="lpw_stable_diffusion_xl",
        add_watermarker=False,
        device_map="balanced" if torch_device.type == "cuda" else None
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    if hasattr(pipe, 'config'):
        pipe.config.clip_skip = clip_skip
    else:
        print("The pipeline does not support clip_skip configuration.")

    pipe.to(torch_device)  # Ensure the entire pipeline is moved to the correct device (GPU or CPU)
    return pipe

def load_lora_weights(pipe, lora_paths, alphas=None, device="cuda"):
    """
    Load and apply multiple LoRA weights sequentially to the pipeline on the correct device.
    """
    if alphas is None:
        alphas = [1.0] * len(lora_paths)

    # Ensure LoRA weights are loaded onto the same device as the model
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    for i, lora_path in enumerate(lora_paths):
        # Load LoRA weights on the same device (GPU/CPU)
        lora_weights = load_safetensors(lora_path, device=device) if lora_path.endswith('.safetensors') else torch.load(lora_path, map_location=device)

        for key, value in lora_weights.items():
            if key in pipe.unet.state_dict():
                # Apply LoRA weights on the correct device
                pipe.unet.state_dict()[key].copy_(pipe.unet.state_dict()[key] * (1 - alphas[i]) + value.to(device) * alphas[i])
        
        print(f"LoRA weights from {lora_path} successfully loaded into the pipeline with alpha {alphas[i]}.")

    return pipe
