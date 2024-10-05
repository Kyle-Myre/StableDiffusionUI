from diffusers    import EulerAncestralDiscreteScheduler
from diffusers    import StableDiffusionXLPipeline
from transformers import BertTokenizer , utils
from rich         import print
import torch , uuid , os

class Utils:
    def generate_filename(self):
        return f'{uuid.uuid4()}.png'

class Tokenizer:
    tokenzier_repo_url = 'bert-base-uncased'
    def tokenize_prompt(self , prompt:str , length:int):
        tokenizer = BertTokenizer.from_pretrained(self.tokenzier_repo_url)
        tokens    = tokenizer.tokenize(prompt)
        if len(tokens) > length:
            print(f"Prompt too long. Truncating from {len(tokens)} tokens to {length} tokens.")
            tokens = tokens[:length]
        return tokenizer.convert_tokens_to_string(tokens)

class Pipeline:
    repo_url:str
    pipe:StableDiffusionXLPipeline
    def __init__(self , repo_url:str) -> None:
        utils.move_cache()
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        self.repo_url = repo_url

    def pipeline(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.repo_url , 
            torch_dtype=torch.float16 , 
            use_safetensors=True,
            device_map="balanced"
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
    

class Generator:
    def generate(self , diffuser:Pipeline , propmt:str , negative:str , dimensions:tuple , guidance:int , steps:int):
        torch.cuda.empty_cache()
        
        result = diffuser.pipe(
        prompt=propmt,
        negative_prompt=negative,
        width=dimensions[0],
        heigth=dimensions[1],
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_images_per_prompt=1
        ).images[0]
        filename = os.path.join('output' , Utils().generate_filename())
        try:
            os.mkdir('output')
        except Exception:
            ...
        finally:
            result.save(filename)

        torch.cuda.empty_cache()
        return filename