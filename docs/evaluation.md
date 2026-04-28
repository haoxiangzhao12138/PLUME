# Evaluation

Evaluation uses the bundled VLM2Vec eval engine in `VLM2Vec` (`eval_twomode.py`). The PLUME eval script wraps it with latent reasoning and MoE support.

## 1. Prerequisites

Install the Python dependencies:

```bash
pip install -r VLM2Vec/requirements.txt
```

Ensure your checkpoint directory contains `preprocessor_config.json` and `chat_template.json`. If missing, copy them from the base model:

```bash
cp /path/to/Qwen2-VL-2B/preprocessor_config.json /path/to/checkpoint/
cp /path/to/Qwen2-VL-2B/chat_template.json /path/to/checkpoint/
```

## 2. Run Evaluation

```bash
cd PLUME

CHECKPOINT=/path/to/checkpoint \
DATA_BASEDIR=/path/to/vlm2vec_eval/MMEB-V2 \
bash scripts/eval_plume_moe.sh
```

`EVAL_ENGINE_DIR` is optional. If omitted, `scripts/eval_plume_moe.sh` uses the bundled engine at `VLM2Vec`. Set `EVAL_ENGINE_DIR=/path/to/VLM2Vec` only when you intentionally want to override the bundled engine.

### Evaluate Specific Modalities or Datasets

```bash
# Image only
MODALITIES="image" \
CHECKPOINT=/path/to/checkpoint \
DATA_BASEDIR=/path/to/vlm2vec_eval/MMEB-V2 \
bash scripts/eval_plume_moe.sh

# Single dataset
MODALITIES="image" \
DATASET_NAMES="ImageNet-1K" \
CHECKPOINT=/path/to/checkpoint \
DATA_BASEDIR=/path/to/vlm2vec_eval/MMEB-V2 \
bash scripts/eval_plume_moe.sh
```

### Key Eval Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EVAL_ENGINE_DIR` | `VLM2Vec` | Path to VLM2Vec repo containing `eval_twomode.py` |
| `CHECKPOINT` | - | Model checkpoint path |
| `DATA_BASEDIR` | - | Root of MMEB-V2 eval data |
| `MODALITIES` | `image,video,visdoc` | Comma-separated modalities to evaluate |
| `DATASET_NAMES` | (all) | Comma-separated datasets to filter |
| `PLUME_LATENT_STEPS` | `auto` | Latent steps (`auto` reads from `trainer_state.json`) |
| `USE_PLUME_LATENT_MOE` | `True` | Enable latent MoE at inference |
| `NPROC_PER_NODE` | `8` | Number of GPUs |
| `HF_DATASETS_OFFLINE` | - | Set to `1` to use cached datasets only |

## 3. Analyze Results

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
