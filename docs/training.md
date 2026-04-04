# Training

All training scripts are in `scripts/` and should be run from the PLUME root directory.

## 1. Single-Node Training

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

### With DeepSpeed ZeRO-3

```bash
USE_DEEPSPEED=1 \
DEEPSPEED_CFG=configs/deepspeed/zero3.json \
MODEL_PATH=/path/to/model \
ANNOTATION_PATH=/path/to/annotations.jsonl \
MEDIA_ROOT=/path/to/media_root \
bash scripts/train_singlenode.sh
```

## 2. Multi-Node Training

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

## 3. Key Training Arguments

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

## 4. Data Format

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
