from diffusers import StableDiffusionXLPipeline , EulerAncestralDiscreteScheduler
from safetensors.torch import load_file as load_safetensors
from transformers.utils import move_cache
from transformers import BertModel
import torch

move_cache()

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

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    if hasattr(pipe, 'config'):
        pipe.config.clip_skip = clip_skip
    else:
        print("The pipeline does not support clip_skip configuration.")
    return pipe

def load_lora_weights(pipe, lora_paths , alphas=1):
    """
    Load and apply multiple LoRA weights sequentially to the pipeline.
    Args:
        pipe: The Stable Diffusion pipeline.
        lora_paths: A list of paths to LoRA models (.safetensors files or repos).
        alphas: Optional list of scaling factors for blending LoRAs. If None, equal scaling is applied.
    """
    if alphas is None:
        alphas = [1.0] * len(lora_paths)
    for i, lora_path in enumerate(lora_paths):
        lora_weights = load_safetensors(lora_path) if lora_path.endswith('.safetensors') else torch.load(lora_path)
        for key, value in lora_weights.items():
            if key in pipe.unet.state_dict():
                pipe.unet.state_dict()[key].copy_(pipe.unet.state_dict()[key] * (1 - alphas[i]) + value * alphas[i])
        print(f"LoRA weights from {lora_path} successfully loaded into the pipeline with alpha {alphas[i]}.")
    return pipe