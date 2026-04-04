# Environment Setup

## 1. Create Conda Environment

```bash
conda create -n plume python=3.10 -y
conda activate plume
```

## 2. Install PyTorch

```bash
# CUDA 12.1 example (adjust for your CUDA version)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

## 3. Install Dependencies

```bash
pip install transformers>=4.46.0
pip install deepspeed>=0.14.0
pip install peft>=0.12.0
pip install accelerate>=0.34.0
pip install datasets
pip install wandb
pip install qwen-vl-utils
```

## 4. Install Flash Attention (recommended)

```bash
pip install flash-attn --no-build-isolation
```

## Requirements Summary

- Python >= 3.10
- PyTorch >= 2.1
- transformers >= 4.46
- deepspeed >= 0.14
- peft >= 0.12
- flash-attn (recommended)
- datasets
- qwen-vl-utils
