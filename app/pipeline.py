from transformers.utils import move_cache
import safetensors.torch

move_cache()

from diffusers import StableDiffusionXLPipeline , EulerAncestralDiscreteScheduler
import torch

def configure_gpu():
    """
    Configure GPU , Only supported when you got CUDA
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

def configure_pipeline(repo_url , device='cuda', clip_skip=2):
    """
    Configure Pipeline and download a model from hugging face if one doesn't exists.
    """
    dtype = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        repo_url,
        torch_dtype=dtype,
        use_safetensors=True,
        custom_pipeline="lpw_stable_diffusion_xl",
        add_watermarker=False,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if device == 'cuda':
        configure_gpu()
        pipe.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("Unsupported device type. Please use 'cuda'.")

    if hasattr(pipe, 'config'):
        pipe.config.clip_skip = clip_skip
    else:
        print("The pipeline does not support clip_skip configuration.")

    pipe.to(device)
    return pipe

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