# StableDiffusion Web UI

This Jupyter notebook provides a setup and workflow for running Stable Diffusion, a state-of-the-art text-to-image generation model. It integrates Hugging Face's `diffusers` library along with various other dependencies to streamline the generation process. Additionally, the notebook supports database operations and visualization, making it versatile for various use cases.

## Requirements

Before running this notebook, ensure you have the following installed:

- Python 3.x
- PyTorch (CUDA 12.1)
- Hugging Face libraries (`transformers`, `accelerate`, `diffusers`, `peft`)
- Google Colab (Optional, for cloud-based execution)
- Additional Python libraries (Matplotlib, Pillow, Rich, SQLAlchemy, IPython-SQL)

### Installation

The required libraries can be installed via pip. The first cell of the notebook contains the necessary commands:

```python
!pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

!pip install git+https://github.com/huggingface/transformers
!pip install git+https://github.com/huggingface/accelerate
!pip install git+https://github.com/huggingface/diffusers
!pip install git+https://github.com/huggingface/peft

!pip install huggingface_hub
!pip install matplotlib pillow rich

!pip install ipython-sql
!pip install sqlalchemy
```

## Usage

### 1. Initialization
- Import the necessary libraries and dependencies.
- Mount your Google Drive if you're using Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

- Set your root path for data and model storage:

```python
root_path = "/content/drive/MyDrive/StableDiffusion/"
```

### 2. Stable Diffusion Setup
- Utilize Hugging Face's `diffusers` library to load the `StableDiffusionXLPipeline`.
- Customize the scheduler using `EulerAncestralDiscreteScheduler`.

### 3. Image Generation
- Generate images based on prompts using the Stable Diffusion pipeline.
- Save and display generated images.

### 4. Database Integration
- Use `SQLAlchemy` and `IPython-SQL` for database operations.
- Store generated results and metadata for easy retrieval.

### 5. Visualization
- Utilize Matplotlib and Pillow for image visualization and manipulation.
- Rich library can be used to enhance console output.

## Notes
- Ensure your GPU is properly configured with CUDA support for faster processing.
- This notebook is optimized for execution in Google Colab but can be adapted for local execution as well.

## License
This project is licensed under the MIT License.

---

Feel free to customize this README further based on any additional specifics you may want to highlight.