# PLUME: Latent Reasoning for Vision-Language Embedding

PLUME is a latent reasoning training and evaluation pipeline for Vision-Language Models (VLMs). It implements curriculum-based continuous thought token training with contrastive learning and Mixture-of-Experts, built on top of Qwen2-VL / Qwen2.5-VL.

## Features

- **Curriculum-based latent reasoning** -- gradually replaces chain-of-thought text with continuous thought (`<ct>`) tokens across training stages
- **Contrastive learning** -- cross-device bidirectional contrastive loss for query/positive embedding alignment
- **Latent MoE** -- optional Mixture-of-Experts transition layer in the latent reasoning loop
- **Gradient checkpointing** -- manual activation checkpointing compatible with KV-cache latent loop
- **Multi-node training** -- ready-to-use scripts for distributed training across multiple nodes
- **Multi-modal evaluation** -- MMEB image / video / visdoc evaluation with latent-MoE support

## Directory Structure

```
PLUME/
├── README.md
├── plume/                              # Python package
│   ├── train/                          #   Training
│   │   ├── train_plume.py              #     Core trainer (PlumeTrainer)
│   │   ├── train_plume_gc.py           #     Gradient-checkpointing variant
│   │   ├── latent_moe.py              #     Latent MoE transition module
│   │   └── argument.py                #     Dataclass argument definitions
│   ├── data/                           #   Data processing
│   │   ├── data_plume.py              #     Dataset, collator, curriculum sampler
│   │   └── rope2d.py                  #     2D/3D RoPE index computation
│   └── eval/                           #   Evaluation analysis tools
│       ├── compare_eval_results.py    #     Compare hit@1 between two runs
│       ├── compute_mmeb_image_hit1_avg.py
│       ├── compute_mmeb_video_hit1_avg.py
│       └── analyze_max_tokens.py      #     Analyze token truncation
├── configs/
│   ├── deepspeed/                      #   DeepSpeed configs
│   │   ├── zero2.json
│   │   ├── zero3.json
│   │   └── zero3_offload.json
│   └── eval/                           #   MMEB dataset configs
│       ├── image.yaml                 #     36 image tasks
│       ├── video.yaml                 #     18 video tasks
│       └── visdoc.yaml                #     Visual document tasks
├── scripts/                            #   Shell launchers
│   ├── train_singlenode.sh            #     Single-node training
│   ├── train_multinode.sh             #     Per-node multi-node script
│   ├── launch_multinode.sh            #     SSH-based multi-node orchestrator
│   └── eval_plume_moe.sh             #     Multi-modal evaluation
└── tools/
    └── check_image.py                  #   Data validation utility
```

---

## 1. Environment Setup

### 1.1 Create Conda Environment

```bash
conda create -n plume python=3.10 -y
conda activate plume
```

### 1.2 Install PyTorch

```bash
# CUDA 12.1 example (adjust for your CUDA version)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 Install Dependencies

```bash
pip install transformers>=4.46.0
pip install deepspeed>=0.14.0
pip install peft>=0.12.0
pip install accelerate>=0.34.0
pip install datasets
pip install wandb
pip install qwen-vl-utils
```

### 1.4 Install Flash Attention (recommended)

```bash
pip install flash-attn --no-build-isolation
```

### 1.5 CUDA cublas Fix (if needed)

If you encounter `cublasSgemmStridedBatched` errors (CUDA version mismatch between system and PyTorch), prepend PyTorch's bundled cublas to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cublas.lib; print(nvidia.cublas.lib.__path__[0])'):${LD_LIBRARY_PATH}"
```

---

## 2. Model & Data

### 2.1 Download Model

Download the pre-trained PLUME model weights from HuggingFace:

```bash
# Option 1: git clone (requires git-lfs)
git lfs install
git clone https://huggingface.co/CUDAOUTOFMEMORY/PLUME-Qwen2-VL-2B /path/to/model

# Option 2: huggingface-cli
pip install huggingface_hub
huggingface-cli download CUDAOUTOFMEMORY/PLUME-Qwen2-VL-2B --local-dir /path/to/model
```

The model is based on Qwen2-VL-2B with additional latent MoE transition weights. The checkpoint includes `trainer_state.json` so that evaluation can auto-detect the number of latent steps.

### 2.2 Download Training Data

Training annotations (JSONL) are hosted on HuggingFace:

```bash
huggingface-cli download zhibinlan/UME-sft-train --repo-type dataset --local-dir /path/to/annotations
```

The corresponding images come from MMEB-V2:

```bash
huggingface-cli download TIGER-Lab/MMEB-V2 --repo-type dataset --local-dir /path/to/MMEB-V2
```

Set `ANNOTATION_PATH` to the downloaded JSONL file and `MEDIA_ROOT` to the image root directory when training.

### 2.3 Download Evaluation Data

Evaluation uses the same MMEB-V2 dataset:

```bash
huggingface-cli download TIGER-Lab/MMEB-V2 --repo-type dataset --local-dir /path/to/vlm2vec_eval/MMEB-V2
```

Set `DATA_BASEDIR` to the downloaded directory when running evaluation.

---

## 3. Training

All training scripts are in `scripts/` and should be run from the PLUME root directory.

### 3.1 Single-Node Training

```bash
cd PLUME

MODEL_PATH=/path/to/model \
ANNOTATION_PATH=/path/to/annotations.jsonl \
MEDIA_ROOT=/path/to/media_root \
bash scripts/train_singlenode.sh
```

Override any parameter via environment variables:

```bash
MODEL_PATH=/path/to/model \
ANNOTATION_PATH=/path/to/annotations.jsonl \
MEDIA_ROOT=/path/to/media_root \
NPROC_PER_NODE=4 \
PER_DEVICE_BS=2 \
GRAD_ACC=4 \
LR=1e-5 \
EPOCHS=1 \
SAVE_STEPS=100 \
bash scripts/train_singlenode.sh
```

#### With DeepSpeed ZeRO-3

```bash
USE_DEEPSPEED=1 \
DEEPSPEED_CFG=configs/deepspeed/zero3.json \
MODEL_PATH=/path/to/model \
ANNOTATION_PATH=/path/to/annotations.jsonl \
MEDIA_ROOT=/path/to/media_root \
bash scripts/train_singlenode.sh
```

### 3.2 Multi-Node Training

Configure node IPs and launch from master:

```bash
cd PLUME

export PLUME_NODE_0=192.168.1.100   # master
export PLUME_NODE_1=192.168.1.101   # worker

# Set data paths (exported to all nodes via SSH)
export MODEL_PATH=/path/to/model
export ANNOTATION_PATH=/path/to/annotations.jsonl
export MEDIA_ROOT=/path/to/media_root

bash scripts/launch_multinode.sh
```

### 3.3 Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--plume_curriculum_stages` | `1` | Comma-separated replacement ratios per stage (e.g. `0,0.25,0.5,0.75,1.0`) |
| `--plume_think_segments` | `4` | Number of CoT segments to partition |
| `--plume_ct_tokens_per_segment` | `1` | Latent `<ct>` tokens per replaced segment |
| `--plume_gen_contrastive_weight` | `1.0` | Generative contrastive loss weight |
| `--plume_disc_contrastive_weight` | `1.0` | Discriminative contrastive loss weight |
| `--plume_contrastive_logit_scale` | `50.0` | Logit scale for contrastive temperature |
| `--plume_annotation_path` | - | Path to JSONL annotations |
| `--plume_media_root` | - | Root directory for media files |
| `--latent_moe_enable` | `True` | Enable latent MoE transition layer |
| `--latent_moe_num_experts` | `4` | Number of routed experts |
| `--latent_moe_top_k` | `2` | Top-k experts per token |
| `--latent_moe_use_shared_expert` | `True` | Add a shared expert (always active) |
| `--latent_moe_context_type` | `none` | MoE context: `none`, `prefix_last`, `disc` |
| `--gradient_checkpointing` | `True` | Manual activation checkpointing |

### 3.4 Data Format

Annotations are JSONL, one JSON object per line with `qry` and `pos` sides:

```json
{
  "qry": {
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe this image."},
      {"from": "gpt", "value": "<think>The image shows a mountain landscape...</think><answer>A scenic mountain view.</answer>"}
    ],
    "image": ["relative/path/to/image.jpg"]
  },
  "pos": {
    "conversations": [
      {"from": "human", "value": "<image>\nWhat does this image depict?"},
      {"from": "gpt", "value": "<think>The photo contains mountains...</think><answer>Mountains at sunset.</answer>"}
    ],
    "image": ["relative/path/to/positive.jpg"]
  },
  "dataset_name": "my_dataset"
}
```

Image paths in `"image"` are relative to `--plume_media_root`.

---

## 4. Evaluation

Evaluation uses the [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) eval engine (`eval_twomode.py`). The PLUME eval script wraps it with latent reasoning and MoE support.

### 4.1 Prerequisites

Clone and install the VLM2Vec eval engine:

```bash
git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git /path/to/VLM2Vec
cd /path/to/VLM2Vec && pip install -e .
```

Ensure your checkpoint directory contains `preprocessor_config.json` and `chat_template.json`. If missing, copy them from the base model:

```bash
cp /path/to/Qwen2-VL-2B/preprocessor_config.json /path/to/checkpoint/
cp /path/to/Qwen2-VL-2B/chat_template.json /path/to/checkpoint/
```

### 4.2 Run Evaluation

```bash
cd PLUME

EVAL_ENGINE_DIR=/path/to/VLM2Vec \
CHECKPOINT=/path/to/checkpoint \
DATA_BASEDIR=/path/to/vlm2vec_eval/MMEB-V2 \
bash scripts/eval_plume_moe.sh
```

#### Evaluate Specific Modalities or Datasets

```bash
# Image only
MODALITIES="image" \
EVAL_ENGINE_DIR=/path/to/VLM2Vec \
CHECKPOINT=/path/to/checkpoint \
DATA_BASEDIR=/path/to/vlm2vec_eval/MMEB-V2 \
bash scripts/eval_plume_moe.sh

# Single dataset
MODALITIES="image" \
DATASET_NAMES="ImageNet-1K" \
EVAL_ENGINE_DIR=/path/to/VLM2Vec \
CHECKPOINT=/path/to/checkpoint \
DATA_BASEDIR=/path/to/vlm2vec_eval/MMEB-V2 \
bash scripts/eval_plume_moe.sh
```

#### Key Eval Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EVAL_ENGINE_DIR` | - | Path to VLM2Vec repo containing `eval_twomode.py` |
| `CHECKPOINT` | - | Model checkpoint path |
| `DATA_BASEDIR` | - | Root of MMEB-V2 eval data |
| `MODALITIES` | `image,video,visdoc` | Comma-separated modalities to evaluate |
| `DATASET_NAMES` | (all) | Comma-separated datasets to filter |
| `PLUME_LATENT_STEPS` | `auto` | Latent steps (`auto` reads from `trainer_state.json`) |
| `USE_PLUME_LATENT_MOE` | `True` | Enable latent MoE at inference |
| `NPROC_PER_NODE` | `8` | Number of GPUs |
| `HF_DATASETS_OFFLINE` | - | Set to `1` to use cached datasets only |

### 4.3 Analyze Results

```bash
# Compute average hit@1 for image tasks
python plume/eval/compute_mmeb_image_hit1_avg.py \
  --score_dir output/Eval/<run>/image-gen

# Compute average hit@1 for video tasks
python plume/eval/compute_mmeb_video_hit1_avg.py \
  --score_dir output/Eval/<run>/video-gen

# Compare two runs
python plume/eval/compare_eval_results.py dir1/ dir2/
```

---

## 5. Requirements

- Python >= 3.10
- PyTorch >= 2.1
- transformers >= 4.46
- deepspeed >= 0.14
- peft >= 0.12
- flash-attn (recommended)
- datasets
- qwen-vl-utils

## License

See the repository root for license information.
