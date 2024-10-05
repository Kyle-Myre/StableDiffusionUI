![[./docs/README.png]](./docs/README.png)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white) ![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black) ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white) ![MIT](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)

This App provides a Interface for running Stable Diffusion,
a state-of-the-art text-to-image generation model. It integrates Hugging Face's
diffusers library along with various other dependencies to streamline the
generation process. Additionally, the App supports database operations
and visualization, making it versatile for various use cases.

Note : This App is not perfect for fast and optimized performance is just an easy abstraction and free option for generating images , you can always download another free option if you want. if you have an idea about how StablDiffusion works i would be happy to listen and learn.

## Project Setup

- Linux (Recomanded)

```bash
python3 -m venv .venv
pip install -r requirements.txt
./runner.local.sh 
```

- Windows

```powershell
py -m venv .venv
pip install -r requirements.txt
./runner.local.bat
```

## Technologies

| **Technology**   | **Description**                                                                 | **Link**                                 |
|------------------|---------------------------------------------------------------------------------|------------------------------------------|
| **Python**       | A high-level programming language used for general-purpose programming.          | [Python](https://www.python.org/)        |
| **CUDA**         | A parallel computing platform and API model created by NVIDIA for GPU computing. | [CUDA](https://developer.nvidia.com/cuda-zone) |
| **Pytorch**      | An open-source machine learning library for deep learning applications.          | [Pytorch](https://pytorch.org/)          |
| **Ubuntu**       | A popular open-source Linux distribution based on Debian.                        | [Ubuntu](https://ubuntu.com/)            |
| **HuggingFace**  | A platform providing open-source tools for machine learning and NLP.             | [HuggingFace](https://huggingface.co/)   |
