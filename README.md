# PLUME: Latent Reasoning Based Universal Multimodal Embedding

[🏡 Project Page](https://haoxiangzhao12138.github.io/PLUME/) | [📄 Paper](https://arxiv.org/abs/2604.02073) | [🤗 Model](https://huggingface.co/CUDAOUTOFMEMORY/PLUME-Qwen2-VL-2B) | [🤗 Training Data](https://huggingface.co/datasets/zhibinlan/UME-sft-train) | [🤗 Eval Data](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2)

PLUME is a latent reasoning framework for universal multimodal embedding (UME). It replaces explicit chain-of-thought (CoT) generation with a short autoregressive rollout of continuous latent states, combined with a semantic-anchor-guided transition adapter (Latent MoE) and a progressive explicit-to-latent curriculum. Built on Qwen2-VL-2B, PLUME achieves **61.6** on the 78-task MMEB-v2 benchmark while delivering **over 30x faster inference** compared to explicit-CoT methods.

<p align="center">
  <img src="https://haoxiangzhao12138.github.io/PLUME/static/images/tradeoff.png" width="50%" alt="Accuracy-Efficiency Tradeoff"/>
</p>

This repository is the official implementation of the paper [PLUME: Latent Reasoning Based Universal Multimodal Embedding](https://arxiv.org/abs/2604.02073).

## 📰 News

- **[2026/04]** Paper released on [arXiv](https://arxiv.org/abs/2604.02073).
- **[2025]** Code and model weights released.

## 📝 TODO

- [x] **Code Released**: Training and evaluation pipeline.
- [x] **Model Released**: Pre-trained [PLUME-Qwen2-VL-2B](https://huggingface.co/CUDAOUTOFMEMORY/PLUME-Qwen2-VL-2B) weights.
- [x] **Paper**: [arXiv](https://arxiv.org/abs/2604.02073).

## 💡 Highlights

- Replaces hundreds of explicit reasoning tokens with only **8 latent steps**, delivering **30.3x faster** inference
- **61.6** overall on the 78-task MMEB-v2 benchmark, surpassing UME-R1 (60.1) and VLM2Vec-V2 (58.0)
- **Curriculum-based latent reasoning** -- gradually replaces chain-of-thought text with continuous thought (`<ct>`) tokens across training stages
- **Contrastive learning** -- cross-device bidirectional contrastive loss for query/positive embedding alignment
- **Latent MoE** -- Mixture-of-Experts transition layer with 4 routed experts + shared expert in the latent reasoning loop
- **Multi-modal evaluation** -- MMEB image / video / visdoc evaluation with latent-MoE support

## 🏗️ Method

<p align="center">
  <img src="https://haoxiangzhao12138.github.io/PLUME/static/images/method.png" width="90%" alt="PLUME Method Overview"/>
</p>

Overview of PLUME. The bottom panel illustrates the latent rollout process. The top-left panel expands the semantic-anchor-guided transition adapter with shared and specialized experts. The top-right panel shows the progressive explicit-to-latent curriculum.

## 🎞️ Results on MMEB-v2

All methods share the same Qwen2-VL-2B backbone.

| Model | Image | Video | VisDoc | All |
|-------|:-----:|:-----:|:------:|:---:|
| VLM2Vec-V2 | 64.9 | 34.9 | 65.4 | 58.0 |
| UME-R1 | 66.6 | 42.2 | 63.9 | 60.1 |
| **PLUME** | **66.3** | **44.1** | **67.5** | **61.6** |

<p align="center">
  <img src="https://haoxiangzhao12138.github.io/PLUME/static/images/radar.png" width="50%" alt="Per-task Performance Comparison"/>
</p>

Per-task performance comparison on MMEB-v2. PLUME consistently outperforms UME-R1 and single-pass baselines across most sub-tasks.

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
├── VLM2Vec/                            #   Bundled MMEB eval engine with eval_twomode.py
├── tools/
│   └── check_image.py
└── docs/                               #   Documentation
    ├── environment.md
    ├── training.md
    └── evaluation.md
```

## 🫡 Acknowledgements

Many thanks to the code bases from [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) and [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL).

## Citation

If you use this code for your research or project, please cite:

```bibtex
@misc{he2026plumelatentreasoningbased,
      title={PLUME: Latent Reasoning Based Universal Multimodal Embedding},
      author={Chenwei He and Xiangzhao Hao and Tianyu Yang and Yuxiang Ma and Yuheng Jia and Lingxiang Wu and Chaoyang Zhao and Haiyun Guo and Jinqiao Wang},
      year={2026},
      eprint={2604.02073},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.02073},
}
```

## License

See the repository root for license information.
