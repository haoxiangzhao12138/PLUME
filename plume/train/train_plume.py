"""
PLUME latent reasoning trainer.

Example (full-parameter, 8 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    plume/train/train_plume.py \
    --model_name_or_path /path/to/model \
    --output_dir output/PLUME-$(date +%Y-%m-%d-%H-%M-%S) \
    --bf16 --learning_rate 8e-6 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --num_train_epochs 1 --model_max_length 8192 \
    --tune_mm_llm True --tune_mm_mlp True --tune_mm_vision False \
    --plume_annotation_path /path/to/annotations.jsonl \
    --plume_media_root /path/to/media_root \
    --plume_curriculum_stages 0,0.25,0.5,0.75,1.0 \
    --plume_think_segments 4 --plume_ct_tokens_per_segment 1

Example (LoRA fine-tuning):
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    plume/train/train_plume.py \
    --model_name_or_path /path/to/model \
    --output_dir output/PLUME-LoRA-$(date +%Y-%m-%d-%H-%M-%S) \
    --bf16 --learning_rate 2e-4 \
    --use_lora True --lora_r 64 --lora_alpha 128 \
    --plume_annotation_path /path/to/annotations.jsonl
"""
import logging
import os
import pathlib
import re
import sys
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2VLImageProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

torch.serialization.add_safe_globals(
    [ZeroStageEnum, LossScaler, DeepSpeedZeroOptimizer, DeepSpeedZeroOptimizer_Stage3]
)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from plume.data.data_plume import (
    CurriculumBalancedSubsetBatchSampler,
    LazyPlumeSFTDataset,
    make_plume_data_module,
)
from plume.train.argument import DataArguments, ModelArguments, TrainingArguments
from plume.train.latent_moe import LatentMoETransition

try:
    import torch.distributed.nn as dist_nn
except Exception:
    dist_nn = None

local_rank = None
IGNORE_INDEX = -100


def rank0_print(*args):
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(*args)


def _debug_cuda_memory_enabled() -> bool:
    return str(os.environ.get("PLUME_DEBUG_MEMORY", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_rank_str() -> str:
    if dist.is_available() and dist.is_initialized():
        try:
            return f"rank{dist.get_rank()}"
        except Exception:
            pass
    if local_rank is not None:
        return f"local_rank{local_rank}"
    return "rank?"


def _log_cuda_memory(tag: str, **kwargs) -> None:
    if (not _debug_cuda_memory_enabled()) or (not torch.cuda.is_available()):
        return
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
        peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(
            f"[PLUME][MEM][{_get_rank_str()}][{tag}] "
            f"alloc={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB "
            f"peak_alloc={peak_alloc_gb:.2f}GB peak_reserved={peak_reserved_gb:.2f}GB "
            f"free={free_gb:.2f}GB total={total_gb:.2f}GB"
            + (f" {extra}" if extra else ""),
            flush=True,
        )
    except Exception as e:
        print(f"[PLUME][MEM][{_get_rank_str()}][{tag}] failed_to_log={e}", flush=True)


def _is_deepspeed_enabled(training_args) -> bool:
    return bool(getattr(training_args, "deepspeed", None))


def _zero3_gathered_parameters(params):
    if params is None:
        return nullcontext()
    ds_params = [
        p for p in params if torch.is_tensor(p) and hasattr(p, "ds_id")
    ]
    if not ds_params:
        return nullcontext()
    return GatheredParameters(ds_params, modifier_rank=None)


def _build_loss_anchor(model, fallback_loss: torch.Tensor) -> torch.Tensor:
    for param in model.parameters():
        if (not param.requires_grad) or (not torch.is_tensor(param)):
            continue
        with _zero3_gathered_parameters([param]):
            if param.numel() <= 0:
                continue
            return param.reshape(-1)[:1].sum() * 0.0
    return fallback_loss * 0.0


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collect state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for _, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for _, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for _, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def _count_params(module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _safe_log_trainable_block(name: str, module) -> None:
    if module is None:
        rank0_print(f"{name}: module is None")
        return
    trainable, total = _count_params(module)
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    rank0_print(f"{name}: trainable={trainable} / total={total} ({pct:.2f}%)")


def safe_print_trainable_parameters(model, use_lora: bool) -> None:
    visual = getattr(model, "visual", None)
    if visual is not None and hasattr(visual, "print_trainable_parameters"):
        try:
            visual.print_trainable_parameters()
        except Exception as e:
            rank0_print(f"[PLUME] visual.print_trainable_parameters() failed: {e}")
            _safe_log_trainable_block("Vision module", visual)
    else:
        _safe_log_trainable_block("Vision module", visual)

    if use_lora:
        if hasattr(model, "print_trainable_parameters"):
            try:
                model.print_trainable_parameters()
            except Exception as e:
                rank0_print(f"[PLUME] model.print_trainable_parameters() failed: {e}")
                _safe_log_trainable_block("Full model", model)
        else:
            _safe_log_trainable_block("Full model", model)
    else:
        llm = getattr(model, "model", None)
        if llm is not None and hasattr(llm, "print_trainable_parameters"):
            try:
                llm.print_trainable_parameters()
            except Exception as e:
                rank0_print(f"[PLUME] llm.print_trainable_parameters() failed: {e}")
                _safe_log_trainable_block("LLM backbone", llm)
        else:
            _safe_log_trainable_block("LLM backbone", llm)

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Trainable parameters: {total_trainable}")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_lm_head(model):
    model = _unwrap_model(model)
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
        return model.base_model.lm_head
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "lm_head")
    ):
        return model.base_model.model.lm_head
    raise AttributeError("Cannot locate lm_head from model.")


def _get_backbone_model(model):
    """
    Return the decoder backbone (Qwen2VLModel/Qwen2_5_VLModel) used to run
    prefix/latent/suffix passes without materializing unnecessary LM logits.
    """
    raw_model = _unwrap_model(model)
    if hasattr(raw_model, "get_base_model"):
        try:
            raw_model = raw_model.get_base_model()
        except Exception:
            pass
    return raw_model.model if hasattr(raw_model, "model") else raw_model


def _get_latent_moe_module(model):
    raw_model = _unwrap_model(model)
    candidate_modules = []
    if raw_model is not None:
        candidate_modules.append(raw_model)
    if hasattr(raw_model, "get_base_model"):
        try:
            base_model = raw_model.get_base_model()
            if base_model is not None:
                candidate_modules.append(base_model)
        except Exception:
            pass
    backbone_model = _get_backbone_model(model)
    if backbone_model is not None:
        candidate_modules.append(backbone_model)

    seen = set()
    for candidate in candidate_modules:
        if candidate is None:
            continue
        cid = id(candidate)
        if cid in seen:
            continue
        seen.add(cid)
        module = getattr(candidate, "latent_moe_transition", None)
        if module is not None:
            return module
    return None


def _normalize_qwen_position_ids(position_ids: torch.LongTensor) -> torch.LongTensor:
    """
    Normalize per-sample position_ids to Qwen expected shape [3, 1, seq].
    Accepts [1, 3, seq], [3, 1, seq], or [3, seq].
    """
    if position_ids.dim() == 2:
        if position_ids.shape[0] == 3:
            normalized = position_ids.unsqueeze(1)
        elif position_ids.shape[1] == 3:
            normalized = position_ids.transpose(0, 1).unsqueeze(1)
        else:
            raise ValueError(
                f"Unsupported 2D position_ids shape: {tuple(position_ids.shape)}"
            )
    elif position_ids.dim() == 3:
        if position_ids.shape[0] == 3:
            normalized = position_ids
        elif position_ids.shape[1] == 3:
            normalized = position_ids.transpose(0, 1)
        else:
            raise ValueError(
                f"Unsupported 3D position_ids shape: {tuple(position_ids.shape)}"
            )
    else:
        raise ValueError(f"Unsupported position_ids dim: {position_ids.dim()}")

    if normalized.shape[1] != 1:
        raise ValueError(
            "Expected per-sample position_ids with batch dim 1, "
            f"got shape {tuple(normalized.shape)}"
        )
    return normalized


def _infer_compute_dtype(model) -> torch.dtype:
    # Prefer active autocast dtype in Trainer mixed-precision context.
    if torch.is_autocast_enabled():
        try:
            return torch.get_autocast_gpu_dtype()
        except Exception:
            pass

    base_model = _unwrap_model(model)
    try:
        return base_model.model.layers[0].self_attn.q_proj.weight.dtype
    except Exception:
        pass

    emb = getattr(base_model, "get_input_embeddings", lambda: None)()
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.dtype

    lm_head = _get_lm_head(base_model)
    if hasattr(lm_head, "weight"):
        return lm_head.weight.dtype

    return torch.float32


def _cast_past_key_values_dtype(past_key_values, target_dtype: torch.dtype):
    if past_key_values is None:
        return None

    # DynamicCache-style objects in newer transformers.
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for idx in range(len(past_key_values.key_cache)):
            key_tensor = past_key_values.key_cache[idx]
            value_tensor = past_key_values.value_cache[idx]
            if torch.is_tensor(key_tensor) and key_tensor.is_floating_point():
                if key_tensor.dtype != target_dtype:
                    past_key_values.key_cache[idx] = key_tensor.to(dtype=target_dtype)
            if torch.is_tensor(value_tensor) and value_tensor.is_floating_point():
                if value_tensor.dtype != target_dtype:
                    past_key_values.value_cache[idx] = value_tensor.to(dtype=target_dtype)
        return past_key_values

    # Legacy tuple cache.
    if isinstance(past_key_values, (tuple, list)):
        converted_layers = []
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                converted_layers.append(layer)
                continue
            key_tensor, value_tensor = layer[0], layer[1]
            layer_tail = list(layer[2:])
            if torch.is_tensor(key_tensor) and key_tensor.is_floating_point():
                if key_tensor.dtype != target_dtype:
                    key_tensor = key_tensor.to(dtype=target_dtype)
            if torch.is_tensor(value_tensor) and value_tensor.is_floating_point():
                if value_tensor.dtype != target_dtype:
                    value_tensor = value_tensor.to(dtype=target_dtype)
            converted_layers.append((key_tensor, value_tensor, *layer_tail))
        return tuple(converted_layers) if isinstance(past_key_values, tuple) else converted_layers

    return past_key_values


def _extract_last_token_rep(
    hidden_states: torch.Tensor, token_ids: torch.Tensor, target_token_id: int
) -> Optional[torch.Tensor]:
    positions = torch.nonzero(token_ids == target_token_id, as_tuple=False)
    if positions.numel() == 0:
        return None
    last_pos = int(positions[-1].item())
    return hidden_states[last_pos]


def _stack_optional_reps(
    reps: List[Optional[torch.Tensor]],
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    stacked = torch.zeros((len(reps), hidden_size), device=device, dtype=dtype)
    valid = torch.zeros((len(reps),), device=device, dtype=torch.bool)
    for idx, rep in enumerate(reps):
        if rep is None:
            continue
        stacked[idx] = rep.to(device=device, dtype=dtype)
        valid[idx] = True
    return stacked, valid


def _compute_contrastive_debug_stats(
    logits_q2p: torch.Tensor,
    labels: torch.LongTensor,
    anchor_mask: torch.BoolTensor,
    valid_cols_mask: torch.BoolTensor,
) -> Dict[str, float]:
    with torch.no_grad():
        num_anchors = int(anchor_mask.sum().item())
        num_valid_cols = int(valid_cols_mask.sum().item())
        if num_anchors <= 0 or num_valid_cols <= 0:
            return {
                "diag_mean": 0.0,
                "offdiag_mean": 0.0,
                "top1_acc": 0.0,
                "num_anchors": float(num_anchors),
                "num_valid_cols": float(num_valid_cols),
            }

        row_logits = logits_q2p[anchor_mask][:, valid_cols_mask].float()
        row_labels = labels[anchor_mask]

        valid_col_indices = torch.nonzero(valid_cols_mask, as_tuple=False).flatten()
        compact_idx = torch.full(
            (valid_cols_mask.size(0),),
            -1,
            dtype=torch.long,
            device=logits_q2p.device,
        )
        compact_idx[valid_col_indices] = torch.arange(
            valid_col_indices.numel(), device=logits_q2p.device, dtype=torch.long
        )
        compact_labels = compact_idx[row_labels]
        valid_rows = compact_labels.ge(0)
        if not torch.any(valid_rows):
            return {
                "diag_mean": 0.0,
                "offdiag_mean": 0.0,
                "top1_acc": 0.0,
                "num_anchors": float(num_anchors),
                "num_valid_cols": float(num_valid_cols),
            }

        row_logits = row_logits[valid_rows]
        compact_labels = compact_labels[valid_rows]

        row_indices = torch.arange(row_logits.size(0), device=row_logits.device)
        diag_scores = row_logits[row_indices, compact_labels]
        diag_mean = float(diag_scores.mean().item())

        offdiag_cols = row_logits.size(1) - 1
        if offdiag_cols > 0:
            offdiag_sum = row_logits.sum(dim=-1) - diag_scores
            offdiag_mean = float((offdiag_sum / offdiag_cols).mean().item())
        else:
            offdiag_mean = diag_mean

        top1 = row_logits.argmax(dim=-1)
        top1_acc = float((top1 == compact_labels).float().mean().item())

        return {
            "diag_mean": diag_mean,
            "offdiag_mean": offdiag_mean,
            "top1_acc": top1_acc,
            "num_anchors": float(row_logits.size(0)),
            "num_valid_cols": float(num_valid_cols),
        }


def _compute_rep_cosine_stats(
    reps_a: torch.Tensor,
    valid_a: torch.BoolTensor,
    reps_b: torch.Tensor,
    valid_b: torch.BoolTensor,
) -> Dict[str, float]:
    pair_mask = valid_a & valid_b
    valid_pairs = int(pair_mask.sum().item())
    if valid_pairs <= 0:
        return {"cos_mean": 0.0, "cos_std": 0.0, "count": 0.0}

    cos = F.cosine_similarity(reps_a[pair_mask].float(), reps_b[pair_mask].float(), dim=-1)
    cos_mean = float(cos.mean().item())
    cos_std = float(cos.std(unbiased=False).item()) if valid_pairs > 1 else 0.0
    return {"cos_mean": cos_mean, "cos_std": cos_std, "count": float(valid_pairs)}


def _distributed_bidirectional_contrastive_loss(
    qry_reps: torch.Tensor,
    pos_reps: torch.Tensor,
    valid_mask: torch.BoolTensor,
    logit_scale: float,
    cross_device: bool = True,
    local_loss: bool = True,
    return_debug_stats: bool = False,
) -> tuple[torch.Tensor, float, float, Optional[Dict[str, float]]]:
    """
    Contrastive loss with optional cross-device negatives.
    Returns (loss, local_valid_anchor_count, global_valid_pair_count, optional_debug_stats).
    """
    zero = qry_reps.sum() * 0.0
    if qry_reps.numel() == 0:
        return zero, 0.0, 0.0, None

    qry_reps = F.normalize(qry_reps, p=2, dim=-1)
    pos_reps = F.normalize(pos_reps, p=2, dim=-1)

    is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    use_cross_device = is_distributed and cross_device and dist_nn is not None

    if use_cross_device:
        all_qry = torch.cat(dist_nn.all_gather(qry_reps), dim=0)
        all_pos = torch.cat(dist_nn.all_gather(pos_reps), dim=0)

        valid_u8 = valid_mask.to(dtype=torch.uint8)
        gathered_valid = [torch.zeros_like(valid_u8) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_valid, valid_u8)
        all_valid = torch.cat([x.bool() for x in gathered_valid], dim=0)

        if local_loss:
            local_bs = qry_reps.size(0)
            rank = dist.get_rank()
            bs_tensor = torch.tensor([local_bs], device=qry_reps.device, dtype=torch.long)
            bs_gathered = [torch.zeros_like(bs_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(bs_gathered, bs_tensor)
            batch_sizes = [int(x.item()) for x in bs_gathered]
            global_start = sum(batch_sizes[:rank])
            labels = (
                torch.arange(local_bs, device=qry_reps.device, dtype=torch.long)
                + global_start
            )
            anchor_mask = valid_mask & all_valid[labels]
            if not torch.any(anchor_mask):
                return zero, 0.0, float(all_valid.sum().item()), None

            logits_q2p = logit_scale * qry_reps @ all_pos.t()
            logits_p2q = logit_scale * pos_reps @ all_qry.t()
            invalid_cols = ~all_valid
            logits_q2p = logits_q2p.masked_fill(invalid_cols.unsqueeze(0), -1e4)
            logits_p2q = logits_p2q.masked_fill(invalid_cols.unsqueeze(0), -1e4)

            loss_q = F.cross_entropy(logits_q2p[anchor_mask], labels[anchor_mask])
            loss_p = F.cross_entropy(logits_p2q[anchor_mask], labels[anchor_mask])
            debug_stats = None
            if return_debug_stats:
                debug_stats = _compute_contrastive_debug_stats(
                    logits_q2p,
                    labels,
                    anchor_mask,
                    all_valid,
                )
            return (
                (loss_q + loss_p) / 2,
                float(anchor_mask.sum().item()),
                float(all_valid.sum().item()),
                debug_stats,
            )

        labels = torch.arange(all_qry.size(0), device=qry_reps.device, dtype=torch.long)
        if not torch.any(all_valid):
            return zero, float(valid_mask.sum().item()), 0.0, None

        logits_q2p = logit_scale * all_qry @ all_pos.t()
        logits_p2q = logits_q2p.t()
        invalid_cols = ~all_valid
        logits_q2p = logits_q2p.masked_fill(invalid_cols.unsqueeze(0), -1e4)
        logits_p2q = logits_p2q.masked_fill(invalid_cols.unsqueeze(0), -1e4)

        loss_q = F.cross_entropy(logits_q2p[all_valid], labels[all_valid])
        loss_p = F.cross_entropy(logits_p2q[all_valid], labels[all_valid])
        debug_stats = None
        if return_debug_stats:
            debug_stats = _compute_contrastive_debug_stats(
                logits_q2p,
                labels,
                all_valid,
                all_valid,
            )
        return (
            (loss_q + loss_p) / 2,
            float(valid_mask.sum().item()),
            float(all_valid.sum().item()),
            debug_stats,
        )

    # Single-device (or fallback) contrastive.
    if not torch.any(valid_mask):
        return zero, 0.0, 0.0, None

    labels = torch.arange(qry_reps.size(0), device=qry_reps.device, dtype=torch.long)
    logits_q2p = logit_scale * qry_reps @ pos_reps.t()
    logits_p2q = logits_q2p.t()
    invalid_cols = ~valid_mask
    logits_q2p = logits_q2p.masked_fill(invalid_cols.unsqueeze(0), -1e4)
    logits_p2q = logits_p2q.masked_fill(invalid_cols.unsqueeze(0), -1e4)
    loss_q = F.cross_entropy(logits_q2p[valid_mask], labels[valid_mask])
    loss_p = F.cross_entropy(logits_p2q[valid_mask], labels[valid_mask])
    valid_pairs = float(valid_mask.sum().item())
    debug_stats = None
    if return_debug_stats:
        debug_stats = _compute_contrastive_debug_stats(
            logits_q2p,
            labels,
            valid_mask,
            valid_mask,
        )
    return (loss_q + loss_p) / 2, valid_pairs, valid_pairs, debug_stats


class CurriculumStageCallback(TrainerCallback):
    def __init__(self, dataset: LazyPlumeSFTDataset, trainer=None):
        self.dataset = dataset
        self.last_stage = -1
        self._trainer = trainer
        self.stage_count = max(self.dataset.num_curriculum_stages(), 1)

        stage_desc = []
        fractions = list(getattr(self.dataset, "curriculum_fractions", []))
        max_latent_tokens = int(getattr(self.dataset, "max_latent_tokens", 0))
        for stage_idx, ratio in enumerate(fractions):
            latent_tokens = int(round(max_latent_tokens * ratio))
            stage_desc.append(f"{stage_idx}:{ratio:.2f}:{latent_tokens}")
        rank0_print(
            "[PLUME] curriculum stages (raw_idx:ratio:latent_tokens)="
            + ",".join(stage_desc)
        )

    def set_trainer(self, trainer):
        self._trainer = trainer

    def _progress_ratio(self, state, args) -> float:
        if state.max_steps and state.max_steps > 0:
            progress = state.global_step / max(1, state.max_steps)
        elif args.max_steps and args.max_steps > 0:
            progress = state.global_step / max(1, args.max_steps)
        else:
            epoch = float(state.epoch or 0.0)
            progress = epoch / max(float(args.num_train_epochs), 1.0)
        return min(max(progress, 0.0), 0.999999)

    def _infer_stage(self, state, args) -> int:
        return self.dataset.curriculum_stage_for_progress(self._progress_ratio(state, args))

    @staticmethod
    def _reset_optimizer_state(optimizer) -> None:
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if not state:
                    continue
                for val in state.values():
                    if torch.is_tensor(val) and val.is_floating_point():
                        val.zero_()

    def _sync_stage(self, state, args, **kwargs):
        progress = self._progress_ratio(state, args)
        stage = self.dataset.curriculum_stage_for_progress(progress)
        prev_answer_latent = bool(getattr(self.dataset, "answer_latent_active", False))
        stage_changed = stage != self.last_stage

        self.dataset.set_curriculum_stage(stage, progress=progress)
        cur = self.dataset.get_curriculum_state()
        answer_latent_changed = bool(cur.get("answer_latent_active", 0.0)) != prev_answer_latent

        if stage_changed:
            is_initial = self.last_stage == -1
            self.last_stage = stage

            if not is_initial:
                optimizer = kwargs.get("optimizer", None)
                if optimizer is None and self._trainer is not None:
                    optimizer = getattr(self._trainer, "optimizer", None)
                if optimizer is not None:
                    self._reset_optimizer_state(optimizer)
                    rank0_print("[PLUME] optimizer state reset for new curriculum stage")
                else:
                    rank0_print("[PLUME] WARNING: could not locate optimizer to reset state")

            rank0_print(
                f"[PLUME] switch curriculum stage={int(cur['stage']) + 1}/{int(cur['total_stages'])}, "
                f"replace_ratio={cur['replace_ratio']:.2f}, "
                f"latent_tokens={int(cur['latent_tokens'])}/{int(cur['max_latent_tokens'])}, "
                f"answer_latent_active={bool(cur.get('answer_latent_active', 0.0))}, "
                f"progress={float(cur.get('progress', 0.0)):.4f}"
            )
        elif answer_latent_changed:
            rank0_print(
                f"[PLUME] final-stage answer mode switched: "
                f"answer_latent_active={bool(cur.get('answer_latent_active', 0.0))}, "
                f"progress={float(cur.get('progress', 0.0)):.4f}, "
                f"stage={int(cur['stage']) + 1}/{int(cur['total_stages'])}"
            )

    def on_train_begin(self, args, state, control, **kwargs):
        if hasattr(self.dataset, "set_sampler_epoch"):
            self.dataset.set_sampler_epoch(0)
        self._sync_stage(state, args, **kwargs)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.dataset, "set_sampler_epoch"):
            self.dataset.set_sampler_epoch(int(state.epoch or 0))

    def on_step_end(self, args, state, control, **kwargs):
        self._sync_stage(state, args, **kwargs)


class PlumeTrainer(Trainer):
    """
    Implements PLUME 3-stage forward in compute_loss:
      1) prefix to <bot>
      2) latent loop with inputs_embeds=previous hidden
      3) suffix teacher forcing and masked CE
    """

    def __init__(
        self,
        *args,
        gen_emb_token_id: int,
        disc_emb_token_id: int,
        gen_contrastive_weight: float = 1.0,
        disc_contrastive_weight: float = 1.0,
        contrastive_logit_scale: float = 50.0,
        contrastive_cross_device: bool = True,
        contrastive_local_loss: bool = True,
        debug_disc_oracle_pos_from_qry: bool = False,
        latent_moe_enable: bool = False,
        latent_moe_balance_loss_weight: float = 0.0,
        latent_moe_context_type: str = "prefix_last",
        **kwargs,
    ):
        self.gen_emb_token_id = int(gen_emb_token_id)
        self.disc_emb_token_id = int(disc_emb_token_id)
        self.gen_contrastive_weight = float(gen_contrastive_weight)
        self.disc_contrastive_weight = float(disc_contrastive_weight)
        self.contrastive_logit_scale = float(contrastive_logit_scale)
        self.contrastive_cross_device = bool(contrastive_cross_device)
        self.contrastive_local_loss = bool(contrastive_local_loss)
        self.debug_disc_oracle_pos_from_qry = bool(debug_disc_oracle_pos_from_qry)
        self.latent_moe_enable = bool(latent_moe_enable)
        self.latent_moe_balance_loss_weight = float(latent_moe_balance_loss_weight)
        context_type = str(latent_moe_context_type or "prefix_last").strip().lower()
        if context_type not in {"none", "prefix_last", "disc"}:
            raise ValueError(
                "latent_moe_context_type must be one of: none, prefix_last, disc"
            )
        self.latent_moe_context_type = context_type
        self._last_latent_moe_balance_loss = 0.0
        self._last_latent_moe_router_entropy = 0.0
        self._cached_hidden_size: Optional[int] = None
        self._cached_rep_dtype: Optional[torch.dtype] = None
        self._cached_compute_dtype: Optional[torch.dtype] = None
        super().__init__(*args, **kwargs)

    def _get_compute_dtype_cached(self, model) -> torch.dtype:
        if self._cached_compute_dtype is None:
            self._cached_compute_dtype = _infer_compute_dtype(model)
        return self._cached_compute_dtype

    def _get_rep_meta_cached(self, model) -> tuple[int, torch.dtype]:
        if self._cached_hidden_size is None:
            raw_model = _unwrap_model(model)
            model_config = getattr(raw_model, "config", None)
            hidden_size = getattr(model_config, "hidden_size", None)
            if hidden_size is None and model_config is not None:
                hidden_size = getattr(getattr(model_config, "text_config", None), "hidden_size", None)
            if hidden_size is None:
                emb = getattr(raw_model, "get_input_embeddings", lambda: None)()
                emb_weight = getattr(emb, "weight", None)
                with _zero3_gathered_parameters([emb_weight] if emb_weight is not None else None):
                    if emb_weight is None:
                        raise RuntimeError("Cannot infer hidden size for contrastive representation.")
                    hidden_size = int(emb_weight.shape[1])
            self._cached_hidden_size = int(hidden_size)
        if self._cached_rep_dtype is None:
            self._cached_rep_dtype = self._get_compute_dtype_cached(model)
        return self._cached_hidden_size, self._cached_rep_dtype

    def _uses_subset_balanced_sampling(self, train_dataset: Optional[Dataset] = None) -> bool:
        if train_dataset is None:
            train_dataset = self.train_dataset
        strategy = getattr(getattr(train_dataset, "data_args", None), "plume_sampling_strategy", "legacy")
        return str(strategy).strip().lower() == "subset_balanced"

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None
        if self._uses_subset_balanced_sampling(train_dataset):
            rank0_print("[PLUME] Using custom batch sampler (subset_balanced)")
            return None
        data_group = getattr(getattr(train_dataset, "data_args", None), "data_group", False)
        if data_group:
            rank0_print("[PLUME] Using SequentialSampler (data_group=True)")
            return SequentialSampler(train_dataset)
        return RandomSampler(train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator,
                description="training",
            )

        if self._uses_subset_balanced_sampling(train_dataset):
            world_size = max(1, int(getattr(self.args, "world_size", 1)))
            batch_sampler = CurriculumBalancedSubsetBatchSampler(
                dataset=train_dataset,
                batch_size=self._train_batch_size,
                world_size=world_size,
                drop_last=bool(self.args.dataloader_drop_last),
                seed=int(getattr(self.args, "seed", 0)),
            )
            rank0_print(
                f"[PLUME] Using subset_balanced sampler with per_device_batch={self._train_batch_size}, "
                f"world_size={world_size}"
            )
            dataloader_kwargs = {
                "batch_sampler": batch_sampler,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "worker_init_fn": seed_worker,
            }
            if self.args.dataloader_num_workers > 0:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
            dataloader = DataLoader(
                train_dataset,
                **dataloader_kwargs,
            )
            return self.accelerator.prepare(dataloader)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _single_sample_loss(
        self,
        model,
        prefix_input_ids: torch.LongTensor,
        prefix_attention_mask: torch.LongTensor,
        prefix_position_ids: torch.LongTensor,
        suffix_input_ids: torch.LongTensor,
        suffix_attention_mask: torch.LongTensor,
        suffix_position_ids: torch.LongTensor,
        suffix_labels: torch.LongTensor,
        latent_steps: int,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> Optional[tuple]:
        # prefix_input_ids: [1, seq_prefix]
        # latent_steps: N
        # suffix_input_ids: [1, seq_suffix]

        device = prefix_input_ids.device
        lm_head = _get_lm_head(model)
        compute_dtype = self._get_compute_dtype_cached(model)
        current_stage = "init"

        prefix_position_ids = _normalize_qwen_position_ids(prefix_position_ids)
        suffix_position_ids = _normalize_qwen_position_ids(suffix_position_ids)

        # Remove right padding before per-sample forward.
        prefix_valid_len = int(prefix_attention_mask[0].sum().item())
        suffix_valid_len = int(suffix_attention_mask[0].sum().item())
        if prefix_valid_len <= 0 or suffix_valid_len <= 0:
            return None

        prefix_input_ids = prefix_input_ids[:, :prefix_valid_len]
        prefix_attention_mask = prefix_attention_mask[:, :prefix_valid_len]
        prefix_position_ids = prefix_position_ids[:, :, :prefix_valid_len]

        suffix_input_ids = suffix_input_ids[:, :suffix_valid_len]
        suffix_attention_mask = suffix_attention_mask[:, :suffix_valid_len]
        suffix_position_ids = suffix_position_ids[:, :, :suffix_valid_len]
        suffix_labels = suffix_labels[:, :suffix_valid_len]

        model_kwargs = {}
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            model_kwargs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            model_kwargs["video_grid_thw"] = video_grid_thw

        try:
            # 1) Prefix
            # Visual inputs can only be consumed by the CausalLM wrapper forward
            # (it inserts image/video features into text embeddings).
            # For pure-text prefix, use backbone to avoid materializing LM logits.
            raw_model = _unwrap_model(model)
            backbone_model = _get_backbone_model(model)
            has_visual_inputs = any(
                key in model_kwargs
                for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
            )
            _log_cuda_memory(
                "single_sample_start",
                prefix_len=int(prefix_input_ids.shape[1]),
                suffix_len=int(suffix_input_ids.shape[1]),
                latent_steps=int(latent_steps),
                has_visual=bool(has_visual_inputs),
            )
            current_stage = "prefix_forward"
            if has_visual_inputs:
                prefix_out = raw_model(
                    input_ids=prefix_input_ids,
                    attention_mask=prefix_attention_mask,
                    position_ids=prefix_position_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                    **model_kwargs,
                )
                prefix_hidden = prefix_out.hidden_states[-1]
            else:
                prefix_out = backbone_model(
                    input_ids=prefix_input_ids,
                    attention_mask=prefix_attention_mask,
                    position_ids=prefix_position_ids,
                    use_cache=True,
                    return_dict=True,
                )
                prefix_hidden = prefix_out.last_hidden_state
            _log_cuda_memory(
                "after_prefix_forward",
                prefix_len=int(prefix_input_ids.shape[1]),
                image_shape=(tuple(pixel_values.shape) if torch.is_tensor(pixel_values) else None),
                video_shape=(tuple(pixel_values_videos.shape) if torch.is_tensor(pixel_values_videos) else None),
            )
            past_key_values = _cast_past_key_values_dtype(prefix_out.past_key_values, compute_dtype)
            last_hidden = prefix_hidden[:, -1, :]  # [1, D]
            if last_hidden.is_floating_point() and last_hidden.dtype != compute_dtype:
                last_hidden = last_hidden.to(dtype=compute_dtype)
            prefix_last_hidden = last_hidden

            # Extract <disc_emb> from prefix
            disc_rep = _extract_last_token_rep(
                prefix_hidden[0], prefix_input_ids[0], self.disc_emb_token_id
            )

            # --- DEBUG: disc_emb position info (first 20 steps) ---
            if not hasattr(self, "_debug_prefix_step"):
                self._debug_prefix_step = 0
            self._debug_prefix_step += 1
            if self._debug_prefix_step <= 20:
                disc_positions = torch.nonzero(
                    prefix_input_ids[0] == self.disc_emb_token_id, as_tuple=False
                ).flatten()
                gen_positions = torch.nonzero(
                    prefix_input_ids[0] == self.gen_emb_token_id, as_tuple=False
                ).flatten()
                IMAGE_PAD_ID = 151655
                VIDEO_PAD_ID = 151656
                n_img_pad = int((prefix_input_ids[0] == IMAGE_PAD_ID).sum().item())
                n_vid_pad = int((prefix_input_ids[0] == VIDEO_PAD_ID).sum().item())
                pv_shape = model_kwargs.get("pixel_values", None)
                pv_shape = tuple(pv_shape.shape) if pv_shape is not None else None
                ig_shape = model_kwargs.get("image_grid_thw", None)
                ig_shape = tuple(ig_shape.shape) if ig_shape is not None else None
                pvv_shape = model_kwargs.get("pixel_values_videos", None)
                pvv_shape = tuple(pvv_shape.shape) if pvv_shape is not None else None
                rank0_print(
                    f"[DEBUG-PREFIX] sample={self._debug_prefix_step} "
                    f"prefix_len={prefix_input_ids.shape[1]} "
                    f"disc_emb_positions={disc_positions.tolist()} "
                    f"gen_emb_in_prefix={gen_positions.tolist()} "
                    f"disc_rep_is_none={disc_rep is None} "
                    f"has_visual={has_visual_inputs} "
                    f"n_img_pad_in_prefix={n_img_pad} n_vid_pad_in_prefix={n_vid_pad} "
                    f"pixel_values_shape={pv_shape} image_grid_thw_shape={ig_shape} "
                    f"pixel_values_videos_shape={pvv_shape}"
                )

            # IMPORTANT: Qwen2-VL expands <|image_pad|> into multiple tokens.
            # Use kv-cache length as the true processed length.
            if hasattr(past_key_values, "get_seq_length"):
                processed_len = int(past_key_values.get_seq_length())
            else:
                processed_len = int(past_key_values[0][0].shape[2])

            # 2) Latent loop
            latent_steps = int(max(latent_steps, 0))
            latent_moe_module = _get_latent_moe_module(model) if self.latent_moe_enable else None
            if self.latent_moe_enable and latent_moe_module is None:
                raise RuntimeError("latent_moe_enable=True but model has no latent_moe_transition module.")
            balance_losses: List[torch.Tensor] = []
            router_entropies: List[torch.Tensor] = []

            current_stage = "latent_loop"
            for step_idx in range(latent_steps):
                step_hidden = last_hidden
                if latent_moe_module is not None:
                    if self.latent_moe_context_type == "none":
                        router_context = None
                    elif self.latent_moe_context_type == "disc":
                        if disc_rep is not None:
                            router_context = disc_rep.unsqueeze(0) if disc_rep.ndim == 1 else disc_rep
                        else:
                            router_context = prefix_last_hidden
                    else:
                        router_context = prefix_last_hidden
                    step_hidden, moe_aux = latent_moe_module(
                        step_hidden,
                        step_id=step_idx,
                        context=router_context,
                    )
                    if step_hidden.is_floating_point() and step_hidden.dtype != compute_dtype:
                        step_hidden = step_hidden.to(dtype=compute_dtype)
                    if isinstance(moe_aux, dict):
                        balance_loss_t = moe_aux.get("balance_loss", None)
                        entropy_t = moe_aux.get("router_entropy", None)
                        if torch.is_tensor(balance_loss_t):
                            balance_losses.append(balance_loss_t)
                        if torch.is_tensor(entropy_t):
                            router_entropies.append(entropy_t)

                step_inputs_embeds = step_hidden.unsqueeze(1)  # [1, 1, D]
                if step_inputs_embeds.is_floating_point() and step_inputs_embeds.dtype != compute_dtype:
                    step_inputs_embeds = step_inputs_embeds.to(dtype=compute_dtype)
                # In simple latent loop, we don't increment position_ids for the latent step
                # or we do if model relies on absolute cache_position.
                # PLUME uses cache_position to track sequence progress.
                step_cache_position = torch.tensor([processed_len], dtype=torch.long, device=device)
                # Position IDs for latent: we use the next position after prefix
                # For Qwen2-VL RoPE 2D, we typically just need the 1D absolute position if it's text-like
                step_position_ids = torch.full(
                    (3, 1, 1), processed_len, dtype=torch.long, device=device
                )

                step_out = backbone_model(
                    input_ids=None,
                    inputs_embeds=step_inputs_embeds,
                    past_key_values=past_key_values,
                    position_ids=step_position_ids,
                    use_cache=True,
                    return_dict=True,
                    cache_position=step_cache_position,
                )
                past_key_values = _cast_past_key_values_dtype(step_out.past_key_values, compute_dtype)
                last_hidden = step_out.last_hidden_state[:, -1, :]
                if last_hidden.is_floating_point() and last_hidden.dtype != compute_dtype:
                    last_hidden = last_hidden.to(dtype=compute_dtype)
                processed_len += 1
            _log_cuda_memory("after_latent_loop", processed_len=processed_len, latent_steps=latent_steps)

            # 3) Suffix
            current_stage = "suffix_forward"
            suffix_len = suffix_input_ids.shape[1]
            suffix_cache_position = torch.arange(
                processed_len, processed_len + suffix_len, dtype=torch.long, device=device
            )
            suffix_out = backbone_model(
                input_ids=suffix_input_ids,
                attention_mask=suffix_attention_mask,
                position_ids=suffix_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                cache_position=suffix_cache_position,
            )
            suffix_hidden = suffix_out.last_hidden_state
            _log_cuda_memory("after_suffix_forward", suffix_len=suffix_len)

            # Extract <gen_emb> from suffix
            gen_rep = _extract_last_token_rep(
                suffix_hidden[0], suffix_input_ids[0], self.gen_emb_token_id
            )

            # --- DEBUG: gen_emb position in suffix ---
            if self._debug_prefix_step <= 20:
                gen_suffix_positions = torch.nonzero(
                    suffix_input_ids[0] == self.gen_emb_token_id, as_tuple=False
                ).flatten()
                disc_suffix_positions = torch.nonzero(
                    suffix_input_ids[0] == self.disc_emb_token_id, as_tuple=False
                ).flatten()
                rank0_print(
                    f"[DEBUG-SUFFIX] sample={self._debug_prefix_step} "
                    f"suffix_len={suffix_input_ids.shape[1]} "
                    f"gen_emb_positions={gen_suffix_positions.tolist()} "
                    f"disc_emb_in_suffix={disc_suffix_positions.tolist()} "
                    f"gen_rep_is_none={gen_rep is None} "
                    f"latent_steps={latent_steps}"
                )

            # CE Loss on suffix
            # Alignment: first token of suffix_input_ids is predicted by the LAST latent hidden state
            # The logits from suffix_out are shifted by 1 relative to inputs.
            current_stage = "lm_head"
            with _zero3_gathered_parameters(list(lm_head.parameters(recurse=False))):
                first_logits = lm_head(last_hidden).unsqueeze(1)  # [1, 1, V]
                suffix_logits = lm_head(suffix_hidden)
            if suffix_logits.shape[1] > 1:
                logits = torch.cat([first_logits, suffix_logits[:, :-1, :]], dim=1)
            else:
                logits = first_logits
            _log_cuda_memory(
                "after_lm_head",
                logits_shape=tuple(logits.shape),
                logits_gb=f"{(logits.numel() * logits.element_size()) / (1024 ** 3):.2f}",
            )

            # suffix_labels: [1, seq_suffix], IGNORE_INDEX where not supervised
            current_stage = "cross_entropy"
            token_loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                suffix_labels.view(-1),
                reduction="none",
            ).view(1, -1)
            _log_cuda_memory(
                "after_cross_entropy",
                token_loss_shape=tuple(token_loss.shape),
            )

            valid_mask = suffix_labels.ne(IGNORE_INDEX)
            if valid_mask.sum() > 0:
                sample_loss = token_loss[valid_mask].mean()
                valid_tokens = int(valid_mask.sum().item())
            else:
                # If no labels (should not happen in SFT), return dummy zero
                sample_loss = token_loss.sum() * 0.0
                valid_tokens = 0

            moe_balance_loss = sample_loss.new_zeros(())
            moe_router_entropy = sample_loss.new_zeros(())
            if balance_losses:
                moe_balance_loss = torch.stack(balance_losses).mean()
                sample_loss = sample_loss + self.latent_moe_balance_loss_weight * moe_balance_loss
            if router_entropies:
                moe_router_entropy = torch.stack(router_entropies).mean()
            self._last_latent_moe_balance_loss = float(moe_balance_loss.detach().item())
            self._last_latent_moe_router_entropy = float(moe_router_entropy.detach().item())
            _log_cuda_memory("single_sample_done", valid_tokens=valid_tokens)

            return sample_loss, latent_steps, valid_tokens, gen_rep, disc_rep
        except torch.OutOfMemoryError:
            _log_cuda_memory(
                "oom",
                stage=current_stage,
                prefix_len=int(prefix_input_ids.shape[1]),
                suffix_len=int(suffix_input_ids.shape[1]),
                latent_steps=int(latent_steps),
                image_shape=(tuple(pixel_values.shape) if torch.is_tensor(pixel_values) else None),
                video_shape=(tuple(pixel_values_videos.shape) if torch.is_tensor(pixel_values_videos) else None),
            )
            raise

    def _run_side_batch(self, model, side_inputs: Dict[str, torch.Tensor]) -> Dict[str, object]:
        prefix_ids = side_inputs["prefix_input_ids"]
        prefix_attn = side_inputs["prefix_attention_mask"]
        prefix_pos = side_inputs["prefix_position_ids"]

        suffix_ids = side_inputs["suffix_input_ids"]
        suffix_attn = side_inputs["suffix_attention_mask"]
        suffix_pos = side_inputs["suffix_position_ids"]
        suffix_labels = side_inputs["suffix_labels"]

        latent_steps_batch = side_inputs["plume_latent_steps"]

        pixel_values_list = side_inputs.get("pixel_values", None)
        image_grid_thw_list = side_inputs.get("image_grid_thw", None)
        pixel_values_videos_list = side_inputs.get("pixel_values_videos", None)
        video_grid_thw_list = side_inputs.get("video_grid_thw", None)

        batch_size = prefix_ids.shape[0]
        sample_losses: List[torch.Tensor] = []
        total_latent_steps = 0
        total_suffix_tokens = 0
        valid_samples = 0
        total_latent_moe_balance_loss = 0.0
        total_latent_moe_router_entropy = 0.0
        gen_reps: List[Optional[torch.Tensor]] = []
        disc_reps: List[Optional[torch.Tensor]] = []

        for idx in range(batch_size):
            pixel_values = (
                pixel_values_list[idx]
                if isinstance(pixel_values_list, list) and idx < len(pixel_values_list)
                else None
            )
            image_grid_thw = (
                image_grid_thw_list[idx]
                if isinstance(image_grid_thw_list, list) and idx < len(image_grid_thw_list)
                else None
            )
            pixel_values_videos = (
                pixel_values_videos_list[idx]
                if isinstance(pixel_values_videos_list, list)
                and idx < len(pixel_values_videos_list)
                else None
            )
            video_grid_thw = (
                video_grid_thw_list[idx]
                if isinstance(video_grid_thw_list, list) and idx < len(video_grid_thw_list)
                else None
            )

            output = self._single_sample_loss(
                model=model,
                prefix_input_ids=prefix_ids[idx : idx + 1],
                prefix_attention_mask=prefix_attn[idx : idx + 1],
                prefix_position_ids=prefix_pos[idx],
                suffix_input_ids=suffix_ids[idx : idx + 1],
                suffix_attention_mask=suffix_attn[idx : idx + 1],
                suffix_position_ids=suffix_pos[idx],
                suffix_labels=suffix_labels[idx : idx + 1],
                latent_steps=int(latent_steps_batch[idx].item()),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
            if output is None:
                gen_reps.append(None)
                disc_reps.append(None)
                continue

            sample_loss, sample_latent_steps, sample_suffix_tokens, gen_rep, disc_rep = output
            sample_losses.append(sample_loss)
            total_latent_steps += sample_latent_steps
            total_suffix_tokens += sample_suffix_tokens
            valid_samples += 1
            total_latent_moe_balance_loss += float(
                getattr(self, "_last_latent_moe_balance_loss", 0.0)
            )
            total_latent_moe_router_entropy += float(
                getattr(self, "_last_latent_moe_router_entropy", 0.0)
            )
            gen_reps.append(gen_rep)
            disc_reps.append(disc_rep)

        if sample_losses:
            ce_loss = torch.stack(sample_losses).mean()
        else:
            # Fallback dummy loss with grad_fn
            ce_loss = prefix_ids.sum() * 0.0

        return {
            "ce_loss": ce_loss,
            "total_latent_steps": total_latent_steps,
            "total_suffix_tokens": total_suffix_tokens,
            "valid_samples": valid_samples,
            "avg_latent_moe_balance_loss": (
                total_latent_moe_balance_loss / max(valid_samples, 1)
            ),
            "avg_latent_moe_router_entropy": (
                total_latent_moe_router_entropy / max(valid_samples, 1)
            ),
            "gen_reps": gen_reps,
            "disc_reps": disc_reps,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        qry_stats = self._run_side_batch(model, inputs["qry"])
        pos_stats = self._run_side_batch(model, inputs["pos"])

        ce_loss = (qry_stats["ce_loss"] + pos_stats["ce_loss"])
        real_pair = inputs.get("plume_real_pair", None)

        # Ensure ce_loss keeps grad_fn even when rank-local valid samples are empty.
        # Use one small anchor tensor instead of scanning all trainable params
        # (full scans are expensive and fragile under ZeRO-3 partition/offload).
        ce_loss = ce_loss + _build_loss_anchor(model, ce_loss)

        pair_count = len(qry_stats["gen_reps"])
        zero = ce_loss * 0.0
        if pair_count > 0:
            hidden_size, rep_dtype = self._get_rep_meta_cached(model)
            rep_device = ce_loss.device

            if real_pair is None:
                real_pair_mask = torch.ones(
                    (pair_count,), device=rep_device, dtype=torch.bool
                )
            else:
                real_pair_mask = real_pair.to(device=rep_device, dtype=torch.bool)

            gen_qry_tensor, gen_qry_valid = _stack_optional_reps(
                qry_stats["gen_reps"], hidden_size, rep_device, rep_dtype
            )
            gen_pos_tensor, gen_pos_valid = _stack_optional_reps(
                pos_stats["gen_reps"], hidden_size, rep_device, rep_dtype
            )
            gen_valid_mask = real_pair_mask & gen_qry_valid & gen_pos_valid

            disc_qry_tensor, disc_qry_valid = _stack_optional_reps(
                qry_stats["disc_reps"], hidden_size, rep_device, rep_dtype
            )
            disc_pos_tensor, disc_pos_valid = _stack_optional_reps(
                pos_stats["disc_reps"], hidden_size, rep_device, rep_dtype
            )
            disc_valid_mask = real_pair_mask & disc_qry_valid & disc_pos_valid

            # --- DEBUG: disc embedding diagnostics (first 3 steps) ---
            if not hasattr(self, '_debug_disc_step_count'):
                self._debug_disc_step_count = 0
            self._debug_disc_step_count += 1
            if self._debug_disc_step_count <= 5:
                with torch.no_grad():
                    dq_norm = disc_qry_tensor.norm(dim=-1)
                    dp_norm = disc_pos_tensor.norm(dim=-1)
                    dq_normed = F.normalize(disc_qry_tensor, p=2, dim=-1)
                    dp_normed = F.normalize(disc_pos_tensor, p=2, dim=-1)
                    cos_diag = (dq_normed * dp_normed).sum(dim=-1)  # per-pair cosine
                    sim_matrix = dq_normed @ dp_normed.t()
                    gq_norm = gen_qry_tensor.norm(dim=-1)
                    gp_norm = gen_pos_tensor.norm(dim=-1)
                    gq_normed = F.normalize(gen_qry_tensor, p=2, dim=-1)
                    gp_normed = F.normalize(gen_pos_tensor, p=2, dim=-1)
                    gen_cos_diag = (gq_normed * gp_normed).sum(dim=-1)
                    gen_sim_matrix = gq_normed @ gp_normed.t()
                    rank0_print(
                        f"[DEBUG-DISC] step={self._debug_disc_step_count} "
                        f"disc_qry_norm={dq_norm.tolist()} disc_pos_norm={dp_norm.tolist()} "
                        f"disc_cos_diag={cos_diag.tolist()} "
                        f"disc_sim_matrix_diag={sim_matrix.diag().tolist()} "
                        f"disc_sim_matrix_offdiag_mean={((sim_matrix.sum() - sim_matrix.diag().sum()) / max(sim_matrix.numel() - sim_matrix.shape[0], 1)).item():.6f}"
                    )
                    rank0_print(
                        f"[DEBUG-GEN]  step={self._debug_disc_step_count} "
                        f"gen_qry_norm={gq_norm.tolist()} gen_pos_norm={gp_norm.tolist()} "
                        f"gen_cos_diag={gen_cos_diag.tolist()} "
                        f"gen_sim_matrix_diag={gen_sim_matrix.diag().tolist()} "
                        f"gen_sim_matrix_offdiag_mean={((gen_sim_matrix.sum() - gen_sim_matrix.diag().sum()) / max(gen_sim_matrix.numel() - gen_sim_matrix.shape[0], 1)).item():.6f}"
                    )

            if self.debug_disc_oracle_pos_from_qry:
                disc_pos_tensor = disc_qry_tensor.detach()
                disc_pos_valid = disc_qry_valid
                disc_valid_mask = real_pair_mask & disc_qry_valid

            gen_contrastive_loss, gen_local_pairs, gen_global_pairs, _ = (
                _distributed_bidirectional_contrastive_loss(
                    gen_qry_tensor,
                    gen_pos_tensor,
                    gen_valid_mask,
                    self.contrastive_logit_scale,
                    cross_device=self.contrastive_cross_device,
                    local_loss=self.contrastive_local_loss,
                )
            )
            disc_contrastive_loss, disc_local_pairs, disc_global_pairs, _ = (
                _distributed_bidirectional_contrastive_loss(
                    disc_qry_tensor,
                    disc_pos_tensor,
                    disc_valid_mask,
                    self.contrastive_logit_scale,
                    cross_device=self.contrastive_cross_device,
                    local_loss=self.contrastive_local_loss,
                )
            )
        else:
            gen_contrastive_loss = zero
            disc_contrastive_loss = zero
            gen_local_pairs = 0.0
            disc_local_pairs = 0.0
            gen_global_pairs = 0.0
            disc_global_pairs = 0.0

        loss = (
            ce_loss
            + self.gen_contrastive_weight * gen_contrastive_loss
            + self.disc_contrastive_weight * disc_contrastive_loss
        )

        total_valid = qry_stats["valid_samples"] + pos_stats["valid_samples"]
        avg_latent_steps = (
            (qry_stats["total_latent_steps"] + pos_stats["total_latent_steps"]) / max(total_valid, 1)
        )
        avg_suffix_tokens = (
            (qry_stats["total_suffix_tokens"] + pos_stats["total_suffix_tokens"]) / max(total_valid, 1)
        )
        avg_latent_moe_balance_loss = (
            qry_stats["avg_latent_moe_balance_loss"] + pos_stats["avg_latent_moe_balance_loss"]
        ) / 2.0
        avg_latent_moe_router_entropy = (
            qry_stats["avg_latent_moe_router_entropy"] + pos_stats["avg_latent_moe_router_entropy"]
        ) / 2.0
        dataset = getattr(self, "train_dataset", None)
        curriculum_state = None
        if isinstance(dataset, LazyPlumeSFTDataset):
            curriculum_state = dataset.get_curriculum_state()

        logging_steps = int(getattr(self.args, "logging_steps", 0) or 0)
        sync_gradients = bool(getattr(getattr(self, "accelerator", None), "sync_gradients", True))
        next_global_step = int(getattr(getattr(self, "state", None), "global_step", 0)) + (
            1 if sync_gradients else 0
        )
        should_log_now = (
            logging_steps > 0
            and sync_gradients
            and next_global_step > 0
            and next_global_step % logging_steps == 0
            and getattr(self, "_last_plume_log_step", None) != next_global_step
        )
        if should_log_now:
            self._last_plume_log_step = next_global_step
            log_payload = {
                "loss": loss.item(),
                "ce_loss": ce_loss.item(),
                "qry_ce_loss": qry_stats["ce_loss"].item(),
                "pos_ce_loss": pos_stats["ce_loss"].item(),
                "gen_contrastive_loss": gen_contrastive_loss.item(),
                "disc_contrastive_loss": disc_contrastive_loss.item(),
                "avg_latent_steps": avg_latent_steps,
                "avg_suffix_tokens": avg_suffix_tokens,
                "contrastive_pairs_local_gen": gen_local_pairs,
                "contrastive_pairs_local_disc": disc_local_pairs,
                "contrastive_pairs_global_gen": gen_global_pairs,
                "contrastive_pairs_global_disc": disc_global_pairs,
                "debug_disc_oracle_pos_from_qry": float(self.debug_disc_oracle_pos_from_qry),
                "latent_moe_enable": float(self.latent_moe_enable),
                "latent_moe_balance_loss": avg_latent_moe_balance_loss,
                "latent_moe_router_entropy": avg_latent_moe_router_entropy,
            }
            if curriculum_state is not None:
                log_payload.update(
                    {
                        "curriculum_stage": float(curriculum_state["stage"]),
                        "curriculum_total_stages": float(curriculum_state["total_stages"]),
                        "curriculum_ratio": float(curriculum_state["replace_ratio"]),
                        "curriculum_latent_tokens": float(curriculum_state["latent_tokens"]),
                    }
                )
            self.log(log_payload)

        outputs = {
            "loss": loss,
            "ce_loss": ce_loss,
            "qry_ce_loss": qry_stats["ce_loss"],
            "pos_ce_loss": pos_stats["ce_loss"],
            "gen_contrastive_loss": gen_contrastive_loss,
            "disc_contrastive_loss": disc_contrastive_loss,
            "avg_latent_steps": avg_latent_steps,
            "avg_suffix_tokens": avg_suffix_tokens,
            "latent_moe_balance_loss": ce_loss.new_tensor(avg_latent_moe_balance_loss),
            "latent_moe_router_entropy": ce_loss.new_tensor(avg_latent_moe_router_entropy),
        }
        return (loss, outputs) if return_outputs else loss


def initialize_latent_moe(model, model_args: ModelArguments):
    raw_model = _unwrap_model(model)
    if not bool(getattr(model_args, "latent_moe_enable", False)):
        rank0_print("[PLUME][LATENT-MOE] disabled")
        return None

    context_type = str(getattr(model_args, "latent_moe_context_type", "prefix_last")).strip().lower()
    if context_type not in {"none", "prefix_last", "disc"}:
        raise ValueError(
            "latent_moe_context_type must be one of: none, prefix_last, disc"
        )

    module_owner = raw_model
    if hasattr(raw_model, "get_base_model"):
        try:
            module_owner = raw_model.get_base_model()
        except Exception:
            module_owner = raw_model

    embedding_owner = module_owner if hasattr(module_owner, "get_input_embeddings") else raw_model
    hidden_size = int(embedding_owner.get_input_embeddings().weight.shape[1])
    latent_moe = LatentMoETransition(
        hidden_size=hidden_size,
        num_experts=max(int(getattr(model_args, "latent_moe_num_experts", 4)), 1),
        top_k=max(int(getattr(model_args, "latent_moe_top_k", 2)), 1),
        use_shared_expert=bool(getattr(model_args, "latent_moe_use_shared_expert", True)),
        step_embed_max_steps=max(int(getattr(model_args, "latent_moe_step_embed_max_steps", 32)), 1),
        expert_dropout=max(float(getattr(model_args, "latent_moe_expert_dropout", 0.0)), 0.0),
    )
    emb_weight = embedding_owner.get_input_embeddings().weight
    latent_moe = latent_moe.to(dtype=emb_weight.dtype)
    module_owner.latent_moe_transition = latent_moe
    if module_owner is not raw_model:
        raw_model.latent_moe_transition = latent_moe
    rank0_print(
        "[PLUME][LATENT-MOE] enabled: "
        f"num_experts={latent_moe.num_experts}, top_k={latent_moe.top_k}, "
        f"use_shared_expert={latent_moe.use_shared_expert}, "
        f"step_embed_max_steps={latent_moe.step_embed_max_steps}, "
        f"expert_dropout={latent_moe.expert_dropout}, "
        f"context_type={context_type}, "
        f"balance_loss_weight={float(getattr(model_args, 'latent_moe_balance_loss_weight', 0.0))}"
    )
    return latent_moe


def initialize_new_tokens(model, tokenizer, model_name: str, force_reinit_all: bool = False):
    # Detect which tokens already exist in the pretrained vocabulary BEFORE adding.
    existing_vocab = set(tokenizer.get_vocab().keys())
    all_tokens = ["<gen_emb>", "<disc_emb>", "<bot>", "<eot>", "<ct>"]
    truly_new = {t for t in all_tokens if t not in existing_vocab}
    preserved = {t for t in all_tokens if t in existing_vocab}

    if force_reinit_all:
        truly_new = set(all_tokens)
        preserved = set()
        rank0_print("[PLUME][TOKENS] force_reinit_all=True, will overwrite ALL token embeddings")

    added = tokenizer.add_tokens(all_tokens)
    print(f"Added {added} tokens")
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    gen_emb_id = tokenizer.convert_tokens_to_ids("<gen_emb>")
    disc_emb_id = tokenizer.convert_tokens_to_ids("<disc_emb>")
    bot_id = tokenizer.convert_tokens_to_ids("<bot>")
    eot_id = tokenizer.convert_tokens_to_ids("<eot>")
    ct_id = tokenizer.convert_tokens_to_ids("<ct>")

    if "qwen2.5" in model_name.lower():
        right_ref_id = tokenizer.convert_tokens_to_ids("</tool_call>")
        left_ref_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    else:
        right_ref_id = tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
        left_ref_id = tokenizer.convert_tokens_to_ids("<|object_ref_start|>")

    fallback_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if right_ref_id < 0:
        right_ref_id = fallback_id
    if left_ref_id < 0:
        left_ref_id = fallback_id

    # Mapping: token -> (token_id, reference_id_to_copy_from)
    init_plan = {
        "<gen_emb>": (gen_emb_id, right_ref_id),
        "<disc_emb>": (disc_emb_id, left_ref_id),
        "<bot>": (bot_id, left_ref_id),
        "<eot>": (eot_id, right_ref_id),
        "<ct>": (ct_id, right_ref_id),
    }

    embedding_weight = _unwrap_model(model).get_input_embeddings().weight
    use_deepspeed = any(
        isinstance(embedding_weight, cls)
        for cls in (DeepSpeedZeroOptimizer, DeepSpeedZeroOptimizer_Stage3)
    ) if hasattr(embedding_weight, "ds_id") else False

    initialized_tokens = []
    skipped_tokens = []

    def _do_init(emb_w):
        for tok_name, (tok_id, ref_id) in init_plan.items():
            if tok_name in truly_new:
                emb_w[tok_id] = emb_w[ref_id].clone()
                initialized_tokens.append(f"{tok_name}(id={tok_id}) <- ref(id={ref_id})")
            else:
                skipped_tokens.append(f"{tok_name}(id={tok_id}) [pretrained, kept]")

    if use_deepspeed:
        with GatheredParameters([embedding_weight], modifier_rank=0):
            is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
            if is_rank0:
                with torch.no_grad():
                    _do_init(embedding_weight)
    else:
        with torch.no_grad():
            _do_init(embedding_weight)

    # Detailed debug logging
    rank0_print(f"[PLUME][TOKENS] vocab_size_before={len(existing_vocab)}, "
                f"newly_added={added}, vocab_size_after={len(tokenizer)}")
    rank0_print(f"[PLUME][TOKENS] truly_new={sorted(truly_new)}, "
                f"preserved={sorted(preserved)}")
    for msg in initialized_tokens:
        rank0_print(f"[PLUME][TOKENS]   INIT: {msg}")
    for msg in skipped_tokens:
        rank0_print(f"[PLUME][TOKENS]   SKIP: {msg}")
    rank0_print(f"[PLUME][TOKENS] ids: <gen_emb>={gen_emb_id}, <disc_emb>={disc_emb_id}, "
                f"<bot>={bot_id}, <eot>={eot_id}, <ct>={ct_id}")


def resolve_attn_implementation(requested: str) -> str:
    """
    Resolve attention backend with graceful fallback:
    - Prefer requested backend.
    - If FA2 is requested but flash_attn is unavailable, fallback to sdpa.
    """
    if requested != "flash_attention_2":
        return requested

    if importlib.util.find_spec("flash_attn") is not None:
        return requested

    rank0_print(
        "[PLUME] flash_attn is not installed, fallback attn_implementation=sdpa. "
        "Install flash-attn to enable flash_attention_2."
    )
    return "sdpa"


def load_qwen_model_with_fallback(
    model_name_or_path: str,
    cache_dir: Optional[str],
    attn_implementation: str,
    torch_dtype,
):
    """
    Load Qwen2-VL / Qwen2.5-VL model with FA2->SDPA fallback on ImportError.
    """
    model_cls = (
        Qwen2_5_VLForConditionalGeneration
        if "qwen2.5" in model_name_or_path.lower()
        else Qwen2VLForConditionalGeneration
    )

    def _load_model(requested_attn_implementation: str):
        try:
            return model_cls.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                attn_implementation=requested_attn_implementation,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        except ValueError as e:
            err_msg = str(e).lower()
            if "deepspeed zero-3 is not compatible with `low_cpu_mem_usage=true`" in err_msg:
                rank0_print(
                    "[PLUME] Detected ZeRO-3 model load incompatibility with "
                    "low_cpu_mem_usage=True; retry with low_cpu_mem_usage=False."
                )
                return model_cls.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_dir,
                    attn_implementation=requested_attn_implementation,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=False,
                )
            raise

    try:
        return _load_model(attn_implementation), attn_implementation
    except ImportError as e:
        err_msg = str(e).lower()
        if attn_implementation == "flash_attention_2" and "flash_attn" in err_msg:
            rank0_print(
                "[PLUME] Failed to enable flash_attention_2 due to missing flash_attn; "
                "retry with attn_implementation=sdpa."
            )
            fallback_impl = "sdpa"
            model = _load_model(fallback_impl)
            return model, fallback_impl
        raise


def _format_side_batch_stats(side_batch: Dict[str, torch.Tensor], prefix: str) -> str:
    prefix_mask = side_batch.get("prefix_attention_mask", None)
    suffix_mask = side_batch.get("suffix_attention_mask", None)
    latent_steps = side_batch.get("plume_latent_steps", None)

    if not torch.is_tensor(prefix_mask) or not torch.is_tensor(suffix_mask):
        return f"{prefix}(unavailable)"

    prefix_lens = prefix_mask.sum(dim=1).detach().cpu()
    suffix_lens = suffix_mask.sum(dim=1).detach().cpu()
    stats = (
        f"{prefix}(prefix_max={int(prefix_lens.max().item())}, "
        f"prefix_mean={float(prefix_lens.float().mean().item()):.1f}, "
        f"suffix_max={int(suffix_lens.max().item())}, "
        f"suffix_mean={float(suffix_lens.float().mean().item()):.1f}"
    )
    if torch.is_tensor(latent_steps):
        latent_steps = latent_steps.detach().cpu()
        stats += (
            f", latent_max={int(latent_steps.max().item())}, "
            f"latent_mean={float(latent_steps.float().mean().item()):.2f}"
        )
    stats += ")"
    return stats


def _run_oom_precheck(
    trainer: PlumeTrainer,
    train_dataset: LazyPlumeSFTDataset,
    data_args: DataArguments,
) -> None:
    if not bool(getattr(data_args, "plume_enable_oom_precheck", True)):
        rank0_print("[PLUME][PRECHECK] disabled by plume_enable_oom_precheck=False")
        return

    force_under_ds = str(
        os.environ.get("PLUME_FORCE_OOM_PRECHECK_UNDER_DEEPSPEED", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if _is_deepspeed_enabled(trainer.args) and (not force_under_ds):
        rank0_print(
            "[PLUME][PRECHECK] skipped because DeepSpeed is enabled. "
            "Set PLUME_FORCE_OOM_PRECHECK_UNDER_DEEPSPEED=1 to force precheck."
        )
        return

    if not torch.cuda.is_available():
        rank0_print("[PLUME][PRECHECK] skipped (CUDA unavailable)")
        return

    num_batches = max(int(getattr(data_args, "plume_oom_precheck_batches", 2)), 1)
    num_stages = max(train_dataset.num_curriculum_stages(), 1)
    stage_indices = list(range(num_stages))
    original_stage = int(train_dataset.curriculum_stage)
    model = trainer.model

    # ---- Build probe indices: from specified subsets or fall back to dataloader ----
    precheck_subsets_raw = str(getattr(data_args, "plume_oom_precheck_subsets", "") or "")
    precheck_subset_names = {s.strip() for s in precheck_subsets_raw.split(",") if s.strip()}
    use_subset_sampling = bool(precheck_subset_names)

    probe_indices: list = []
    if use_subset_sampling:
        for dataset_name, indices in getattr(train_dataset, "dataset_to_indices", {}).items():
            if dataset_name in precheck_subset_names:
                probe_indices.extend(indices)

        rank0_print(
            f"[PLUME][PRECHECK] subset sampling: subsets={sorted(precheck_subset_names)}, "
            f"matched={len(probe_indices)}/{len(train_dataset)}"
        )
        if not probe_indices:
            rank0_print(
                "[PLUME][PRECHECK] WARNING: no samples matched precheck subsets, "
                "falling back to dataloader iteration"
            )
            use_subset_sampling = False

    rank0_print(
        f"[PLUME][PRECHECK] start: stages={[s + 1 for s in stage_indices]}/{num_stages}, "
        f"probe_batches_per_stage={num_batches}, "
        f"mode={'subset_random' if use_subset_sampling else 'dataloader'}"
    )

    # Get collator for manual batch construction
    collator = None
    if use_subset_sampling:
        collator = trainer.data_collator

    try:
        for stage_idx in stage_indices:
            train_dataset.set_curriculum_stage(stage_idx)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if use_subset_sampling:
                # Random sample from matched indices, construct batch manually
                bs = max(int(getattr(trainer.args, "per_device_train_batch_size", 1)), 1)
                for probe_idx in range(num_batches):
                    sampled_indices = [
                        probe_indices[i]
                        for i in torch.randint(0, len(probe_indices), (bs,)).tolist()
                    ]
                    instances = []
                    for si in sampled_indices:
                        try:
                            instances.append(train_dataset._get_item(si))
                        except Exception as e:
                            rank0_print(
                                f"[PLUME][PRECHECK] skip sample idx={si}: {e}"
                            )
                    if not instances:
                        rank0_print(
                            f"[PLUME][PRECHECK] no valid samples for stage={stage_idx + 1}, "
                            f"probe={probe_idx + 1}, skipping"
                        )
                        continue

                    batch = collator(instances)
                    _run_single_probe(
                        model, trainer, batch, stage_idx, num_stages, probe_idx, num_batches
                    )
            else:
                # Original dataloader-based iteration
                if hasattr(train_dataset, "set_sampler_epoch"):
                    train_dataset.set_sampler_epoch(stage_idx)
                dataloader = trainer.get_train_dataloader()
                data_iter = iter(dataloader)
                for probe_idx in range(num_batches):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break
                    _run_single_probe(
                        model, trainer, batch, stage_idx, num_stages, probe_idx, num_batches
                    )
    finally:
        train_dataset.set_curriculum_stage(original_stage)
        if hasattr(train_dataset, "set_sampler_epoch"):
            train_dataset.set_sampler_epoch(0)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    rank0_print("[PLUME][PRECHECK] passed")


def _run_single_probe(model, trainer, batch, stage_idx, num_stages, probe_idx, num_batches):
    """Execute a single forward+backward probe and report peak memory."""
    batch_stats = (
        _format_side_batch_stats(batch["qry"], "qry")
        + ", "
        + _format_side_batch_stats(batch["pos"], "pos")
    )
    prepared_batch = trainer._prepare_inputs(batch)

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with trainer.compute_loss_context_manager():
            probe_loss = trainer.compute_loss(model, prepared_batch)
        trainer.accelerator.backward(probe_loss)
        model.zero_grad(set_to_none=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(
                "[PLUME][PRECHECK] OOM before training starts. "
                f"stage={stage_idx + 1}/{num_stages}, "
                f"probe_batch={probe_idx + 1}/{num_batches}, {batch_stats}. "
                "Try reducing per_device_train_batch_size/model_max_length, "
                "or disable DoRA via --lora_use_dora False."
            ) from e
        raise

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    rank0_print(
        f"[PLUME][PRECHECK] ok: stage={stage_idx + 1}/{num_stages}, "
        f"probe_batch={probe_idx + 1}/{num_batches}, "
        f"peak_alloc={peak_gb:.2f}GB, {batch_stats}"
    )


def train(attn_implementation: str = "flash_attention_2"):
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    training_args.data_group = data_args.data_group
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Real-time log to output_dir
    log_file = os.path.join(training_args.output_dir, f"train_rank{local_rank}.log")
    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                try:
                    f.write(data)
                    f.flush()
                except Exception:
                    pass
        def flush(self):
            for f in self.files:
                try:
                    f.flush()
                except Exception:
                    pass
    try:
        _log_fh = open(log_file, "a", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, _log_fh)
        sys.stderr = _Tee(sys.__stderr__, _log_fh)
    except Exception as e:
        rank0_print(f"[PLUME] Failed to init log file {log_file}: {e}")

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    if training_args.gradient_checkpointing:
        rank0_print(
            "[PLUME] gradient_checkpointing=True is incompatible with latent KV-cache loop. "
            "Disable gradient checkpointing."
        )
        training_args.gradient_checkpointing = False

    attn_implementation = resolve_attn_implementation(model_args.attn_implementation)
    model, attn_implementation = load_qwen_model_with_fallback(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    rank0_print(f"[PLUME] Using attn_implementation={attn_implementation}")

    if "qwen2.5" in model_args.model_name_or_path.lower():
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path
        ).image_processor
    else:
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    initialize_new_tokens(model, tokenizer, model_args.model_name_or_path)
    set_model(model_args, model)
    model.config.use_cache = True

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(","),
            lora_dropout=model_args.lora_dropout,
            init_lora_weights="gaussian",
            use_dora=bool(model_args.lora_use_dora),
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        # IMPORTANT: In PLUME, we added new tokens like <bot>, <eot>, <gen_emb>.
        # We MUST ensure the input embeddings and lm_head are trainable even in LoRA mode.
        _unwrap_model(model).get_input_embeddings().requires_grad_(True)
        _get_lm_head(model).requires_grad_(True)

    initialize_latent_moe(model, model_args)

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        safe_print_trainable_parameters(model, use_lora=model_args.use_lora)

    data_module = make_plume_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        enable_lazy_tokenization=(training_args.dataloader_num_workers > 0),
    )
    stage_callback = CurriculumStageCallback(data_module["train_dataset"])
    train_dataset = data_module["train_dataset"]
    global_batch = (
        int(training_args.per_device_train_batch_size)
        * max(1, int(training_args.gradient_accumulation_steps))
        * max(1, int(training_args.world_size))
    )
    rank0_print("[PLUME][TRAIN] ===== Training Setup =====")
    rank0_print(f"[PLUME][TRAIN] train_samples={len(train_dataset)}")
    rank0_print(
        f"[PLUME][TRAIN] per_device_batch={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, world_size={training_args.world_size}, "
        f"effective_global_batch={global_batch}"
    )
    rank0_print(
        f"[PLUME][TRAIN] max_steps={training_args.max_steps}, "
        f"num_train_epochs={training_args.num_train_epochs}, "
        f"learning_rate={training_args.learning_rate}"
    )
    rank0_print(
        f"[PLUME][TRAIN] contrastive_weights(gen/disc)="
        f"{data_args.plume_gen_contrastive_weight}/{data_args.plume_disc_contrastive_weight}, "
        f"logit_scale={data_args.plume_contrastive_logit_scale}, "
        f"debug_disc_oracle_pos_from_qry={data_args.plume_debug_disc_oracle_pos_from_qry}"
    )
    rank0_print(
        f"[PLUME][TRAIN] sampling_strategy={data_args.plume_sampling_strategy}, "
        f"data_group={data_args.data_group}, "
        f"final_stage_portion={data_args.plume_final_stage_portion}, "
        f"latent_answer_in_final_half={data_args.plume_latent_answer_in_final_half}, "
        f"final_stage_answer_portion={data_args.plume_final_stage_answer_portion}"
    )
    rank0_print(
        f"[PLUME][TRAIN] lora: enabled={model_args.use_lora}, r={model_args.lora_r}, "
        f"alpha={model_args.lora_alpha}, dropout={model_args.lora_dropout}, "
        f"use_dora={model_args.lora_use_dora}"
    )
    rank0_print(
        f"[PLUME][TRAIN] deepspeed_enabled={_is_deepspeed_enabled(training_args)}, "
        f"deepspeed_cfg={getattr(training_args, 'deepspeed', None)}"
    )
    rank0_print(
        f"[PLUME][TRAIN] latent_moe: enabled={model_args.latent_moe_enable}, "
        f"num_experts={model_args.latent_moe_num_experts}, top_k={model_args.latent_moe_top_k}, "
        f"use_shared_expert={model_args.latent_moe_use_shared_expert}, "
        f"context_type={model_args.latent_moe_context_type}, "
        f"balance_w={model_args.latent_moe_balance_loss_weight}"
    )
    rank0_print("[PLUME][TRAIN] =========================")
    gen_emb_token_id = tokenizer.convert_tokens_to_ids("<gen_emb>")
    disc_emb_token_id = tokenizer.convert_tokens_to_ids("<disc_emb>")
    trainer = PlumeTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        gen_emb_token_id=gen_emb_token_id,
        disc_emb_token_id=disc_emb_token_id,
        gen_contrastive_weight=data_args.plume_gen_contrastive_weight,
        disc_contrastive_weight=data_args.plume_disc_contrastive_weight,
        contrastive_logit_scale=data_args.plume_contrastive_logit_scale,
        contrastive_cross_device=data_args.plume_contrastive_cross_device,
        contrastive_local_loss=data_args.plume_contrastive_local_loss,
        debug_disc_oracle_pos_from_qry=data_args.plume_debug_disc_oracle_pos_from_qry,
        latent_moe_enable=model_args.latent_moe_enable,
        latent_moe_balance_loss_weight=model_args.latent_moe_balance_loss_weight,
        latent_moe_context_type=model_args.latent_moe_context_type,
        callbacks=[stage_callback],
        **data_module,
    )
    stage_callback.set_trainer(trainer)
    _run_oom_precheck(trainer, train_dataset, data_args)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

