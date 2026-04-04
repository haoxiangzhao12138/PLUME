# PLUME: Latent Reasoning for Universal Multi-modal Embedding

[🏡 Project Page](https://haoxiangzhao12138.github.io/PLUME/) | [📄 Paper](#) | [🤗 Model](https://huggingface.co/CUDAOUTOFMEMORY/PLUME-Qwen2-VL-2B) | [🤗 Training Data](https://huggingface.co/datasets/zhibinlan/UME-sft-train) | [🤗 Eval Data](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2)

<!-- TODO: Add teaser figure -->
<!-- ![PLUME](assets/teaser.png) -->

This repository is the official implementation of the paper [PLUME: Latent Reasoning for Universal Multi-modal Embedding](#).

## 📰 News

- **[2025]** Code and model weights released.

## 📝 TODO

- [x] **Code Released**: Training and evaluation pipeline.
- [x] **Model Released**: Pre-trained [PLUME-Qwen2-VL-2B](https://huggingface.co/CUDAOUTOFMEMORY/PLUME-Qwen2-VL-2B) weights.
- [ ] **Paper**: Coming soon.

## 💡 Highlights

- **Curriculum-based latent reasoning** -- gradually replaces chain-of-thought text with continuous thought (`<ct>`) tokens across training stages
- **Contrastive learning** -- cross-device bidirectional contrastive loss for query/positive embedding alignment
- **Latent MoE** -- Mixture-of-Experts transition layer with 4 routed experts + shared expert in the latent reasoning loop
- **Multi-modal evaluation** -- MMEB image / video / visdoc evaluation with latent-MoE support

## 📦 Model & Data

| Resource | Link | Description |
|---|---|---|
| Model weights | [CUDAOUTOFMEMORY/PLUME-Qwen2-VL-2B](https://huggingface.co/CUDAOUTOFMEMORY/PLUME-Qwen2-VL-2B) | Pre-trained PLUME model (Qwen2-VL-2B + Latent MoE) |
| Training annotations | [zhibinlan/UME-sft-train](https://huggingface.co/datasets/zhibinlan/UME-sft-train) | JSONL annotations for training |
| Images & eval data | [TIGER-Lab/MMEB-V2](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2) | Multi-modal images for training and evaluation |

## 🔧 Getting Started

- [Environment Setup](docs/environment.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)

## 📁 Directory Structure

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
│       ├── compare_eval_results.py
│       ├── compute_mmeb_image_hit1_avg.py
│       ├── compute_mmeb_video_hit1_avg.py
│       └── analyze_max_tokens.py
├── configs/
│   ├── deepspeed/                      #   DeepSpeed configs (zero2/zero3/offload)
│   └── eval/                           #   MMEB dataset configs (image/video/visdoc)
├── scripts/                            #   Shell launchers
│   ├── train_singlenode.sh
│   ├── train_multinode.sh
│   ├── launch_multinode.sh
│   └── eval_plume_moe.sh
├── tools/
│   └── check_image.py
└── docs/                               #   Documentation
    ├── environment.md
    ├── training.md
    └── evaluation.md
```

## 🎞️ Results

<!-- TODO: Add results table and figures -->

## 🫡 Acknowledgements

Many thanks to the code bases from [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) and [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL).

## Citation

If you use this code for your research or project, please cite:

```latex
@article{plume2025,
  title={PLUME: Latent Reasoning for Universal Multi-modal Embedding},
  author={},
  year={2025},
  url={https://github.com/haoxiangzhao12138/PLUME}
}
```

## License

See the repository root for license information.
