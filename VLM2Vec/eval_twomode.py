import datetime
import logging
import json
import random
import time
import re

import numpy as np
import os
import pickle
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
'''
cd /home/guohaiyun/yangtianyu/UME-R1 && \
export prefix=/home/guohaiyun/yangtianyu && \
export MODEL_NAME=/home/guohaiyun/yangtianyu/UME-R1/output/UME-R1-2B-Coconut-smoke-1step-2026-02-14-15-48-27/checkpoint-300 && \
export MODEL_BASE=/home/share/yty_model/UME-R1/2B/UME-R1/2B && \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
MODE=gen \
DATASET_NAMES=CIRR \
DATA_BASEDIR=/home/share/yty_data/vlm2vec_eval/MMEB-V2 \
USE_COCONUT_LATENT_REASONING=True \
COCONUT_LATENT_STEPS=2 \
COCONUT_PREFIX_TEXT='<think><bot>' \
COCONUT_FORCED_SUFFIX_TEXT='<eot></think><answer>' \
OUTPUT_SUFFIX='image-cirr-gen-coconut-ls2-ckpt300' \
bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_cirr_twomode.sh

export prefix=/home/guohaiyun/yangtianyu
export MODEL_NAME="$prefix/UME-R1/output/UME-R1-2B-Coconut-GC-LargeBatch-fix-2026-02-26-22-24-13/checkpoint-110"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_NAME="$MODEL_NAME" \
MODEL_BACKBONE="qwen2_vl" \
MODEL_BASE="/home/share/yty_model/UME-R1/2B/UME-R1/2B" \
USE_COCONUT_LATENT_REASONING=True \
COCONUT_LATENT_STEPS=8 \
bash "$prefix/UME-R1/src/eval/VLM2Vec/experiments/public/eval/eval_coconut_cirr_twomode.sh"
'''

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_flashatt
from datasets import Dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
from src.eval_utils.metrics import RankingMetrics
from src.model.model import MMEBModel
from src.model.latent_moe import LatentMoETransition
from src.model.processor import get_backbone_name, load_processor, COLPALI, MODEL2BACKBONE
from src.utils import batch_to_device, print_rank, print_master
import multiprocessing
from multiprocessing import Pool, cpu_count
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)

def get_embedding_idx(generated_ids_trimmed, EMBEDDING_TOKEN_ID):

    embedding_idx = []
    # Search from the last token forward
    for i, out_ids in enumerate(generated_ids_trimmed):
        embed_exist = False
        for j in range(len(out_ids) - 1, -1, -1):
            if out_ids[j] == EMBEDDING_TOKEN_ID:
                if j + 1 >= len(out_ids) - 1:
                    embedding_idx.append(-1)
                else:
                    embedding_idx.append(j + 1)
                embed_exist = True
                break
        if not embed_exist:
            embedding_idx.append(-1)
        
        # embedding_idx.append(-1)

    return embedding_idx

def normalize_reps(reps):
    # Normalize the representations
    reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps


def _is_qwen25_model(model_args: ModelArguments, model_path: str) -> bool:
    if "qwen2_5" in str(model_args.model_backbone).lower() or "qwen2.5" in str(model_args.model_backbone).lower():
        return True
    return "qwen2.5" in str(model_path).lower()


def _resolve_model_paths(model_args: ModelArguments) -> Tuple[str, str]:
    """
    Returns:
        - base_model_path: full model path to load before applying adapter
        - adapter_path: None for full-model inference, otherwise LoRA adapter path
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
    adapter_exists = os.path.isfile(adapter_config_path)
    use_adapter = bool(model_args.lora or adapter_exists)
    if not use_adapter:
        return model_name_or_path, None

    base_model_path = model_args.model_base
    if (not base_model_path) and adapter_exists:
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_model_path = adapter_cfg.get("base_model_name_or_path", None)
    if not base_model_path:
        raise ValueError(
            "LoRA evaluation requires base model path. "
            "Please pass --model_base, or ensure adapter_config.json contains base_model_name_or_path."
        )
    return base_model_path, model_name_or_path


def _normalize_latent_moe_state_key(raw_key: str) -> Optional[str]:
    marker = "latent_moe_transition."
    idx = raw_key.find(marker)
    if idx < 0:
        return None
    return raw_key[idx + len(marker) :]


def _iter_model_weight_files(model_path: str) -> List[str]:
    if not os.path.isdir(model_path):
        return []
    pattern = re.compile(
        r"^(model(?:-\d{5}-of-\d{5})?|pytorch_model(?:-\d{5}-of-\d{5})?)\.(safetensors|bin)$"
    )
    files = []
    for name in sorted(os.listdir(model_path)):
        if pattern.match(name):
            files.append(os.path.join(model_path, name))
    return files


def _extract_latent_moe_state_from_file(weight_path: str) -> Dict[str, torch.Tensor]:
    extracted: Dict[str, torch.Tensor] = {}
    if weight_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except Exception:
            safe_open = None

        if safe_open is None:
            return extracted

        with safe_open(weight_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                normalized = _normalize_latent_moe_state_key(key)
                if normalized is not None:
                    extracted[normalized] = f.get_tensor(key)
        return extracted

    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return extracted
    for key, value in state.items():
        normalized = _normalize_latent_moe_state_key(str(key))
        if normalized is not None and torch.is_tensor(value):
            extracted[normalized] = value
    return extracted


def _load_latent_moe_state_from_checkpoint(model, model_path: str) -> bool:
    latent_moe = _get_latent_moe_module(model)
    if latent_moe is None:
        return False

    moe_state: Dict[str, torch.Tensor] = {}
    for weight_file in _iter_model_weight_files(model_path):
        moe_state.update(_extract_latent_moe_state_from_file(weight_file))

    if not moe_state:
        return False

    missing, unexpected = latent_moe.load_state_dict(moe_state, strict=False)
    print_master(
        f"[EVAL][LATENT-MOE] loaded from {model_path}; "
        f"keys={len(moe_state)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )
    return True


def _parse_latent_moe_excluded_experts(raw: str, num_experts: int) -> Tuple[int, ...]:
    """Parse comma-separated expert indices (0..num_experts-1) to exclude from routing."""
    text = str(raw or "").strip()
    if not text:
        return ()
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part, 10))
    seen = sorted(set(out))
    for e in seen:
        if e < 0 or e >= num_experts:
            raise ValueError(
                f"coconut_latent_moe_exclude_experts: index {e} out of range [0, {num_experts})."
            )
    return tuple(seen)


def _attach_eval_latent_moe(
    model,
    eval_args: Optional["EvalArguments"],
    tokenizer=None,
    checkpoint_path: Optional[str] = None,
) -> None:
    if eval_args is None:
        return
    if not bool(getattr(eval_args, "use_coconut_latent_moe", False)):
        return

    raw_model = _unwrap_model(model)
    hidden_size = int(raw_model.get_input_embeddings().weight.shape[1])
    num_experts = max(int(getattr(eval_args, "coconut_latent_moe_num_experts", 4)), 1)
    top_k = max(int(getattr(eval_args, "coconut_latent_moe_top_k", 2)), 1)
    excluded = _parse_latent_moe_excluded_experts(
        str(getattr(eval_args, "coconut_latent_moe_exclude_experts", "") or ""),
        num_experts,
    )
    if num_experts - len(excluded) < top_k:
        raise ValueError(
            f"coconut_latent_moe_exclude_experts={list(excluded)} leaves only "
            f"{num_experts - len(excluded)} routable experts but top_k={top_k}."
        )
    latent_moe = LatentMoETransition(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        use_shared_expert=bool(getattr(eval_args, "coconut_latent_moe_use_shared_expert", True)),
        step_embed_max_steps=max(int(getattr(eval_args, "coconut_latent_moe_step_embed_max_steps", 32)), 1),
        excluded_expert_indices=excluded if excluded else None,
    )
    emb_weight = raw_model.get_input_embeddings().weight
    latent_moe = latent_moe.to(dtype=emb_weight.dtype)
    raw_model.latent_moe_transition = latent_moe

    loaded = False
    if checkpoint_path:
        loaded = _load_latent_moe_state_from_checkpoint(model, checkpoint_path)
    if not loaded:
        raise ValueError(
            "[EVAL][LATENT-MOE] enabled but no latent_moe_transition weights were found in checkpoint. "
            "Please point --model_name/--checkpoint_path to a MoE checkpoint."
        )

    context_type = str(getattr(eval_args, "coconut_latent_moe_context_type", "prefix_last")).strip().lower()
    if context_type not in {"none", "prefix_last", "disc"}:
        raise ValueError("coconut_latent_moe_context_type must be one of: none, prefix_last, disc")

    disc_emb_id = -1
    if tokenizer is not None:
        disc_emb_id = int(tokenizer.convert_tokens_to_ids("<disc_emb>"))
    excl_str = (
        ",".join(str(i) for i in excluded)
        if excluded
        else "(none)"
    )
    print_master(
        "[EVAL][LATENT-MOE] enabled: "
        f"num_experts={getattr(eval_args, 'coconut_latent_moe_num_experts', 4)}, "
        f"top_k={getattr(eval_args, 'coconut_latent_moe_top_k', 2)}, "
        f"use_shared_expert={getattr(eval_args, 'coconut_latent_moe_use_shared_expert', True)}, "
        f"step_embed_max_steps={getattr(eval_args, 'coconut_latent_moe_step_embed_max_steps', 32)}, "
        f"exclude_experts={excl_str}, "
        f"context_type={context_type}, disc_emb_id={disc_emb_id}"
    )


def _load_qwen_generation_model(
    model_args: ModelArguments,
    processor: AutoProcessor,
    eval_args: Optional["EvalArguments"] = None,
):
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    base_model_path, adapter_path = _resolve_model_paths(model_args)
    is_qwen25 = _is_qwen25_model(model_args, base_model_path)
    model_cls = Qwen2_5_VLForConditionalGeneration if is_qwen25 else Qwen2VLForConditionalGeneration
    model = model_cls.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # COCONUT checkpoints add new special tokens. Ensure embeddings match tokenizer size.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        vocab_size = len(processor.tokenizer)
        if model.get_input_embeddings().weight.shape[0] != vocab_size:
            print_master(
                f"[EVAL] Resize token embeddings: "
                f"{model.get_input_embeddings().weight.shape[0]} -> {vocab_size}"
            )
            model.resize_token_embeddings(vocab_size)

    if adapter_path is not None:
        from peft import PeftModel

        print_master(f"[EVAL] Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            torch_dtype=torch.bfloat16,
            is_trainable=False,
        )
        # Merge for faster/easier multi-GPU eval with plain HF model forward/generate.
        model = model.merge_and_unload()

    tokenizer = processor.tokenizer if (processor is not None and hasattr(processor, "tokenizer")) else None
    _attach_eval_latent_moe(
        model=model,
        eval_args=eval_args,
        tokenizer=tokenizer,
        checkpoint_path=model_name_or_path,
    )

    return model

def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size


def _resolve_task_data_path(base_dir: str, rel_path: str, key_name: str) -> str:
    """
    Resolve dataset path under --data_basedir with backward-compatible fallbacks.
    MMEB-V2 image files are typically under image-tasks/<dataset>, while some
    legacy configs still use image-tasks/MMEB/<dataset>.
    """
    if not rel_path:
        return rel_path
    resolved = os.path.join(base_dir, rel_path)
    if os.path.exists(resolved):
        return resolved

    candidates = []
    if key_name == "image_root":
        normalized_rel = rel_path.replace("image-tasks/MMEB/", "image-tasks/")
        normalized_rel = normalized_rel.replace("image-tasks/MMEB", "image-tasks")
        if normalized_rel != rel_path:
            candidates.append(os.path.join(base_dir, normalized_rel))

    for cand in candidates:
        if os.path.exists(cand):
            print_master(
                f"[EVAL] path fallback for {key_name}: {resolved} -> {cand}"
            )
            return cand
    return resolved


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_latent_moe_module(model):
    raw_model = _unwrap_model(model)
    return getattr(raw_model, "latent_moe_transition", None)


def _extract_last_token_rep(
    hidden_states: torch.Tensor, input_ids: torch.Tensor, token_id: int
) -> Optional[torch.Tensor]:
    positions = torch.nonzero(input_ids == int(token_id), as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    return hidden_states[int(positions[-1].item())]


def _safe_dist_barrier(timeout_minutes: int = 30) -> None:
    if not dist.is_initialized():
        return
    # Use plain barrier without device_ids — the device_ids parameter
    # is only meaningful for the Gloo backend and can cause hangs with NCCL.
    dist.barrier()


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
    raw_model = _unwrap_model(model)
    if hasattr(raw_model, "get_base_model"):
        try:
            raw_model = raw_model.get_base_model()
        except Exception:
            pass
    return raw_model.model if hasattr(raw_model, "model") else raw_model


def _infer_compute_dtype(model) -> torch.dtype:
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


def _latent_reasoning_generate_rep(
    model,
    inputs: Dict[str, torch.Tensor],
    tokenizer,
    max_new_tokens: int,
    latent_steps: int,
    prefix_token_ids: List[int],
    forced_suffix_token_ids: List[int],
    gen_emb_token_id: int,
    use_latent_moe: bool = False,
    latent_moe_context_type: str = "prefix_last",
    disc_emb_token_id: int = -1,
    local_rank: int = 0,
    debug_log_tokens: bool = False,
    collect_latent_trace: bool = False,
) -> Tuple[torch.Tensor, bool, Optional[Dict[str, object]]]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    if input_ids.shape[0] != 1:
        raise ValueError(
            "Latent reasoning generation currently requires per_device_eval_batch_size=1."
        )
    device = input_ids.device

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)

    valid_len = int(attention_mask[0].sum().item())
    prefix_input_ids = input_ids[:, :valid_len]
    prefix_attention_mask = attention_mask[:, :valid_len]

    # Feed assistant prefix to <think><bot> (or user-defined prefix).
    if prefix_token_ids:
        prefix_append = torch.tensor(
            prefix_token_ids, dtype=prefix_input_ids.dtype, device=device
        ).unsqueeze(0)
        has_prefix_tail = (
            prefix_input_ids.shape[1] >= prefix_append.shape[1]
            and torch.equal(prefix_input_ids[:, -prefix_append.shape[1] :], prefix_append)
        )
        if not has_prefix_tail:
            prefix_input_ids = torch.cat([prefix_input_ids, prefix_append], dim=1)
            prefix_attention_mask = torch.cat(
                [
                    prefix_attention_mask,
                    torch.ones(
                        (1, prefix_append.shape[1]), dtype=prefix_attention_mask.dtype, device=device
                    ),
                ],
                dim=1,
            )

    model_kwargs = {}
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        value = inputs.get(key, None)
        if value is not None:
            model_kwargs[key] = value

    raw_model = _unwrap_model(model)
    backbone_model = _get_backbone_model(model)
    lm_head = _get_lm_head(model)
    compute_dtype = _infer_compute_dtype(model)

    # 1) Prefix forward to <think><bot>.
    prefix_out = raw_model(
        input_ids=prefix_input_ids,
        attention_mask=prefix_attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        **model_kwargs,
    )
    prefix_hidden = prefix_out.hidden_states[-1]
    past_key_values = _cast_past_key_values_dtype(prefix_out.past_key_values, compute_dtype)
    last_hidden = prefix_hidden[:, -1, :]
    if last_hidden.is_floating_point() and last_hidden.dtype != compute_dtype:
        last_hidden = last_hidden.to(dtype=compute_dtype)
    prefix_last_hidden = last_hidden

    if hasattr(past_key_values, "get_seq_length"):
        processed_len = int(past_key_values.get_seq_length())
    else:
        processed_len = int(past_key_values[0][0].shape[2])

    latent_moe_context_type = str(latent_moe_context_type or "prefix_last").strip().lower()
    if latent_moe_context_type not in {"none", "prefix_last", "disc"}:
        raise ValueError("latent_moe_context_type must be one of: none, prefix_last, disc")
    latent_moe_module = _get_latent_moe_module(model) if use_latent_moe else None
    if use_latent_moe and latent_moe_module is None:
        raise RuntimeError("use_latent_moe=True but model has no latent_moe_transition module.")
    disc_rep = None
    if use_latent_moe and latent_moe_context_type == "disc" and int(disc_emb_token_id) >= 0:
        disc_rep = _extract_last_token_rep(
            prefix_hidden[0], prefix_input_ids[0], int(disc_emb_token_id)
        )

    semantic_anchor = None
    if collect_latent_trace and int(disc_emb_token_id) >= 0:
        semantic_anchor = _extract_last_token_rep(
            prefix_hidden[0], prefix_input_ids[0], int(disc_emb_token_id)
        )

    latent_trace_list: List[torch.Tensor] = [] if collect_latent_trace else []

    # 2) Latent loop: inputs_embeds = previous hidden, with KV-cache reuse.
    latent_steps = int(max(latent_steps, 0))
    for step_idx in range(latent_steps):
        step_hidden = last_hidden
        if latent_moe_module is not None:
            if latent_moe_context_type == "none":
                router_context = None
            elif latent_moe_context_type == "disc":
                if disc_rep is not None:
                    router_context = disc_rep.unsqueeze(0) if disc_rep.ndim == 1 else disc_rep
                else:
                    router_context = prefix_last_hidden
            else:
                router_context = prefix_last_hidden
            step_hidden, _ = latent_moe_module(
                step_hidden,
                step_id=step_idx,
                context=router_context,
            )
            if step_hidden.is_floating_point() and step_hidden.dtype != compute_dtype:
                step_hidden = step_hidden.to(dtype=compute_dtype)

        step_inputs_embeds = step_hidden.unsqueeze(1)
        if step_inputs_embeds.is_floating_point() and step_inputs_embeds.dtype != compute_dtype:
            step_inputs_embeds = step_inputs_embeds.to(dtype=compute_dtype)
        step_cache_position = torch.tensor([processed_len], dtype=torch.long, device=device)
        step_position_ids = torch.full((3, 1, 1), processed_len, dtype=torch.long, device=device)
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
        if collect_latent_trace:
            latent_trace_list.append(last_hidden.detach().float().squeeze(0).cpu())

    generated_token_ids: List[int] = []
    generated_hidden_states: List[torch.Tensor] = []

    # 3) Force-append "<eot></think><answer>".
    if forced_suffix_token_ids:
        suffix_input_ids = torch.tensor(
            forced_suffix_token_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        suffix_len = suffix_input_ids.shape[1]
        suffix_attention_mask = torch.ones((1, suffix_len), dtype=torch.long, device=device)
        suffix_position_ids = (
            torch.arange(processed_len, processed_len + suffix_len, dtype=torch.long, device=device)
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )
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
        past_key_values = _cast_past_key_values_dtype(suffix_out.past_key_values, compute_dtype)
        suffix_hidden = suffix_out.last_hidden_state
        for pos in range(suffix_len):
            generated_token_ids.append(int(suffix_input_ids[0, pos].item()))
            generated_hidden_states.append(suffix_hidden[0, pos, :])
        last_hidden = suffix_hidden[:, -1, :]
        if last_hidden.is_floating_point() and last_hidden.dtype != compute_dtype:
            last_hidden = last_hidden.to(dtype=compute_dtype)
        processed_len += suffix_len

    eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, (tuple, list, set)):
        eos_token_ids = {int(x) for x in eos_token_id if x is not None}
    elif eos_token_id is None:
        eos_token_ids = set()
    else:
        eos_token_ids = {int(eos_token_id)}

    # 4) Continue normal autoregressive decoding for answer + <gen_emb>.
    stopped_by_special = False
    for _ in range(int(max_new_tokens)):
        next_token_logits = lm_head(last_hidden).float()
        next_token_id = int(torch.argmax(next_token_logits[0], dim=-1).item())

        token_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        token_position_ids = torch.full((3, 1, 1), processed_len, dtype=torch.long, device=device)
        token_cache_position = torch.tensor([processed_len], dtype=torch.long, device=device)
        token_out = backbone_model(
            input_ids=token_input_ids,
            past_key_values=past_key_values,
            position_ids=token_position_ids,
            use_cache=True,
            return_dict=True,
            cache_position=token_cache_position,
        )
        past_key_values = _cast_past_key_values_dtype(token_out.past_key_values, compute_dtype)
        token_hidden = token_out.last_hidden_state[:, -1, :]
        if token_hidden.is_floating_point() and token_hidden.dtype != compute_dtype:
            token_hidden = token_hidden.to(dtype=compute_dtype)
        last_hidden = token_hidden
        processed_len += 1

        generated_token_ids.append(next_token_id)
        generated_hidden_states.append(token_hidden.squeeze(0))

        if next_token_id == int(gen_emb_token_id):
            # Training uses the hidden state at <gen_emb> itself.
            # Stop here to keep inference aligned with training-time rep extraction.
            stopped_by_special = True
            break
        if next_token_id in eos_token_ids:
            stopped_by_special = True
            break

    # Reached the max decode budget without observing <gen_emb>/eos.
    reached_max_new_tokens = int(max_new_tokens) > 0 and (not stopped_by_special)

    # === DEBUG: Log token generation details (only when debug_log_tokens is enabled) ===
    if debug_log_tokens and (reached_max_new_tokens or (local_rank == 0 and random.random() < 0.01)):
        try:
            # Decode generated tokens
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
            prefix_text = tokenizer.decode(prefix_token_ids, skip_special_tokens=False) if prefix_token_ids else ""
            suffix_text = tokenizer.decode(forced_suffix_token_ids, skip_special_tokens=False) if forced_suffix_token_ids else ""

            logger.info("=" * 80)
            logger.info(f"[COCONUT LATENT DEBUG] Rank {local_rank}")
            logger.info(f"  Latent steps: {latent_steps}")
            logger.info(f"  Prefix text: {prefix_text}")
            logger.info(f"  Suffix text: {suffix_text}")
            logger.info(f"  Generated tokens count: {len(generated_token_ids)}")
            logger.info(f"  Max new tokens: {max_new_tokens}")
            logger.info(f"  Reached max: {reached_max_new_tokens}")
            logger.info(f"  Stopped by special: {stopped_by_special}")
            logger.info(f"  Gen_emb token ID: {gen_emb_token_id}")
            logger.info(f"  EOS token IDs: {eos_token_ids}")
            logger.info(f"  Generated text:\n{generated_text}")
            logger.info(f"  Last 20 token IDs: {generated_token_ids[-20:]}")

            # Check if gen_emb exists in generated tokens
            has_gen_emb = gen_emb_token_id in generated_token_ids
            has_eos = any(tid in eos_token_ids for tid in generated_token_ids)
            logger.info(f"  Has <gen_emb>: {has_gen_emb}")
            logger.info(f"  Has EOS: {has_eos}")
            logger.info("=" * 80)
        except Exception as e:
            logger.warning(f"Failed to log token details: {e}")

    def _build_latent_trace(rep_tensor: torch.Tensor) -> Dict[str, object]:
        hid_dim = int(rep_tensor.shape[-1])
        if latent_trace_list:
            latent_hiddens = torch.stack(latent_trace_list, dim=0)
        else:
            latent_hiddens = torch.empty(0, hid_dim, dtype=torch.float32)
        disc_cpu = None
        if semantic_anchor is not None:
            disc_cpu = semantic_anchor.detach().float().cpu()
        return {
            "latent_hiddens": latent_hiddens,
            "disc_anchor": disc_cpu,
            "gen_hidden": rep_tensor.detach().float().cpu(),
        }

    if not generated_hidden_states:
        rep_pre = prefix_hidden[0, -1, :]
        rep_norm = normalize_reps(rep_pre)
        if collect_latent_trace:
            return rep_norm, reached_max_new_tokens, _build_latent_trace(rep_pre)
        return rep_norm, reached_max_new_tokens, None

    # Align with training: use hidden state at the last <gen_emb> token.
    embedding_idx = -1
    for idx in range(len(generated_token_ids) - 1, -1, -1):
        if int(generated_token_ids[idx]) == int(gen_emb_token_id):
            embedding_idx = idx
            break

    if embedding_idx < 0:
        rep = generated_hidden_states[-1]
    else:
        rep = generated_hidden_states[embedding_idx]
    rep_norm = normalize_reps(rep)
    if collect_latent_trace:
        return rep_norm, reached_max_new_tokens, _build_latent_trace(rep)
    return rep_norm, reached_max_new_tokens, None


def encode_embeddings(
    model: AutoModelForCausalLM,
    loader: DataLoader,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    full_dataset: Dataset,
    encode_side: str,
    description: str = "Encoding",
    processor: AutoProcessor = None,
    mode: str = "gen",
    eval_args: Optional["EvalArguments"] = None,
    valid_num_rows: Optional[int] = None,
) -> tuple[np.ndarray, list, int]:
    """
    Encodes embeddings for a given dataset using the model, handling both standard and
    late-interaction models in a DDP-safe manner.
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Check if the model is a late-interaction type
    is_late_interaction = (model_args.model_backbone == COLPALI)

    local_embeds = []
    local_gt_infos = []
    local_hit_max_new_tokens_flags: List[bool] = []
    local_max_len = 0
    use_coconut_latent = bool(
        eval_args is not None
        and mode == "gen"
        and getattr(eval_args, "use_coconut_latent_reasoning", False)
    )
    tokenizer = processor.tokenizer if (processor is not None and hasattr(processor, "tokenizer")) else None
    latent_steps = 0
    prefix_token_ids: List[int] = []
    forced_suffix_token_ids: List[int] = []
    gen_emb_token_id = -1
    use_latent_moe = False
    latent_moe_context_type = "prefix_last"
    disc_emb_token_id = -1
    if use_coconut_latent:
        if tokenizer is None:
            raise ValueError("Latent reasoning eval requires processor.tokenizer.")
        latent_steps = int(max(getattr(eval_args, "coconut_latent_steps", 0), 0))
        prefix_token_ids = tokenizer.encode(
            getattr(eval_args, "coconut_prefix_text", "<think><bot>"),
            add_special_tokens=False,
        )
        forced_suffix_token_ids = tokenizer.encode(
            getattr(eval_args, "coconut_forced_suffix_text", "<eot></think><answer>"),
            add_special_tokens=False,
        )
        gen_emb_token_id = int(tokenizer.convert_tokens_to_ids("<gen_emb>"))
        if gen_emb_token_id < 0:
            raise ValueError("Tokenizer does not contain <gen_emb> token.")
        use_latent_moe = bool(getattr(eval_args, "use_coconut_latent_moe", False))
        latent_moe_context_type = str(
            getattr(eval_args, "coconut_latent_moe_context_type", "prefix_last")
        ).strip().lower()
        disc_emb_token_id = int(tokenizer.convert_tokens_to_ids("<disc_emb>"))
        debug_log_tokens = getattr(eval_args, "debug_log_tokens", False)
        if local_rank == 0:
            print_master(
                "[EVAL] enable latent reasoning in gen mode: "
                f"latent_steps={latent_steps}, "
                f"prefix='{getattr(eval_args, 'coconut_prefix_text', '<think><bot>')}', "
                f"forced_suffix='{getattr(eval_args, 'coconut_forced_suffix_text', '<eot></think><answer>')}', "
                f"debug_log_tokens={debug_log_tokens}, "
                f"use_latent_moe={use_latent_moe}, latent_moe_context_type={latent_moe_context_type}"
            )

    model.eval()
    with torch.no_grad():
        for inputs, dataset_info in tqdm(loader, desc=f"{description} (rank {local_rank})", disable=local_rank > 0):
            inputs = batch_to_device(inputs, training_args.device)
            inputs.pop("texts", None)
            inputs.pop("images", None)
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                if mode == "gen":
                    if use_coconut_latent:
                        # Current latent path uses per-sample KV-cache loop.
                        # For multimodal Qwen inputs, enforce batch_size=1 to keep
                        # image token packing aligned and deterministic.
                        if int(inputs["input_ids"].shape[0]) != 1:
                            raise ValueError(
                                "Latent reasoning eval requires per_device_eval_batch_size=1. "
                                f"Got batch_size={int(inputs['input_ids'].shape[0])}."
                            )
                        latent_rep, hit_max_new_tokens, _ = _latent_reasoning_generate_rep(
                            model=model,
                            inputs=inputs,
                            tokenizer=tokenizer,
                            max_new_tokens=model_args.max_new_tokens,
                            latent_steps=latent_steps,
                            prefix_token_ids=prefix_token_ids,
                            forced_suffix_token_ids=forced_suffix_token_ids,
                            gen_emb_token_id=gen_emb_token_id,
                            use_latent_moe=use_latent_moe,
                            latent_moe_context_type=latent_moe_context_type,
                            disc_emb_token_id=disc_emb_token_id,
                            local_rank=local_rank,
                            debug_log_tokens=debug_log_tokens,
                        )
                        output = [latent_rep]
                        local_hit_max_new_tokens_flags.append(bool(hit_max_new_tokens))
                    else:
                        generated_output = model.generate(
                            **inputs,
                            max_new_tokens=model_args.max_new_tokens,
                            output_hidden_states=True,
                            return_dict_in_generate=True,
                            use_cache=True,
                        )
                        generated_ids = generated_output.sequences
                        hidden_states = generated_output.hidden_states
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                        ]
                        embedding_idx = get_embedding_idx(
                            generated_ids_trimmed, processor.tokenizer.get_vocab()["<gen_emb>"]
                        )
                        output = []
                        for i, idx in enumerate(embedding_idx):
                            embedding_reps = hidden_states[idx][-1][i].squeeze(0)
                            embedding_reps = normalize_reps(embedding_reps)
                            output.append(embedding_reps)
                elif mode == "disc":
                    output_forward = model(**inputs, output_hidden_states=True, return_dict=True)
                    hidden_states = output_forward.hidden_states[-1]
                    embedding_idx = get_embedding_idx(
                        inputs["input_ids"], processor.tokenizer.get_vocab()["<disc_emb>"]
                    )
                    output = []
                    for i, idx in enumerate(embedding_idx):
                        embedding_reps = hidden_states[i][idx]
                        embedding_reps = normalize_reps(embedding_reps)
                        output.append(embedding_reps)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                output = torch.stack(output, dim=0)
                reps = output.detach()

            if encode_side == "qry":
                local_gt_infos.extend(dataset_info)
            else:
                local_gt_infos.extend([info["cand_name"] for info in dataset_info])

            if is_late_interaction and reps.dim() == 3:
                local_max_len = max(local_max_len, reps.shape[1])

            local_embeds.append(reps)

    if not local_embeds:
        # Handle cases where a rank gets no data
        return np.array([]), [], 0

    # === DDP Synchronization and Padding for Late-Interaction Models ===
    if is_late_interaction:
        if dist.is_initialized():
            # 1. Find the global maximum sequence length across all ranks
            local_max_len_tensor = torch.tensor(local_max_len, device=training_args.device)
            dist.all_reduce(local_max_len_tensor, op=dist.ReduceOp.MAX)
            global_max_len = local_max_len_tensor.item()
        else:
            global_max_len = local_max_len

        # 2. Pad all local embeddings to the global max length
        padded_embeds = []
        for reps_batch in local_embeds:
            if reps_batch.dim() == 3:
                B, L, H = reps_batch.shape
                padding_size = global_max_len - L
                padded_batch = F.pad(reps_batch, (0, 0, 0, padding_size), "constant", 0)
                padded_embeds.append(padded_batch)
            else: # Should not happen if model is consistently late-interaction
                padded_embeds.append(reps_batch)

        embeds_tensor = torch.cat(padded_embeds, dim=0).contiguous()
    else: # Standard dense models
        embeds_tensor = torch.cat(local_embeds, dim=0).contiguous()

    if dist.is_initialized():
        _safe_dist_barrier()  # Ensure all ranks have completed their local embedding computation
    
    # === Gather embeddings and keys from all ranks ===
    if dist.is_initialized() and full_dataset.num_rows >= world_size:
        print_master(f"Gathering {encode_side} embeddings across all ranks...")

        # Use the more efficient all_gather_into_tensor for tensors
        output_shape = list(embeds_tensor.shape)
        output_shape[0] = full_dataset.num_rows
        embeds_tensor = embeds_tensor.to(training_args.device)
        gathered_embeds_tensor = torch.empty(output_shape, dtype=embeds_tensor.dtype, device=training_args.device)
        dist.all_gather_into_tensor(gathered_embeds_tensor, embeds_tensor)
        final_embeddings = gathered_embeds_tensor.cpu().float().numpy()
        # Gather metadata, for which all_gather_object is appropriate
        gathered_gt_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_gt_infos, local_gt_infos)
        all_gt_infos = [key for rank_keys in gathered_gt_infos for key in rank_keys]
    else:
        all_gt_infos = local_gt_infos
        final_embeddings = embeds_tensor.cpu().float().numpy()

    hit_max_new_tokens_count = 0
    if use_coconut_latent:
        if dist.is_initialized() and full_dataset.num_rows >= world_size:
            gathered_hit_flags = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_hit_flags, local_hit_max_new_tokens_flags)
            all_hit_flags = [flag for rank_flags in gathered_hit_flags for flag in rank_flags]
        else:
            all_hit_flags = local_hit_max_new_tokens_flags

        valid_rows = int(valid_num_rows) if valid_num_rows is not None else len(all_hit_flags)
        valid_rows = min(valid_rows, len(all_hit_flags))
        hit_max_new_tokens_count = int(sum(1 for flag in all_hit_flags[:valid_rows] if flag))
        hit_rate = (hit_max_new_tokens_count / valid_rows * 100) if valid_rows > 0 else 0
        print_master(
            f"[EVAL] {description}: hit max_new_tokens without <gen_emb>/eos = "
            f"{hit_max_new_tokens_count}/{valid_rows} ({hit_rate:.2f}%)"
        )
        if hit_max_new_tokens_count > 0:
            print_master(
                f"[WARNING] {hit_max_new_tokens_count} samples reached max_new_tokens! "
                f"This may indicate the model is not generating <gen_emb> token properly. "
                f"Check the debug logs above for token sequences."
            )

    return final_embeddings, all_gt_infos, hit_max_new_tokens_count

@dataclass
class EvalArguments:
    """
    Arguments for evaluation, including model, data, and training configurations.
    """
    qry_mode: str = field(default="gen", metadata={"help": "Mode of qry embedding, gen for generative embedding, disc for discriminative embedding."})
    tgt_mode: str = field(default="gen", metadata={"help": "Mode of target embedding, gen for generative embedding, disc for discriminative embedding."})
    dataset_names: str = field(
        default="",
        metadata={
            "help": "Comma-separated dataset keys from yaml to evaluate (e.g., CIRR,MSCOCO). Empty means all."
        },
    )
    use_coconut_latent_reasoning: bool = field(
        default=False,
        metadata={
            "help": "Enable COCONUT latent reasoning inference for gen mode."
        },
    )
    coconut_latent_steps: int = field(
        default=4,
        metadata={
            "help": "Latent loop steps N used in COCONUT inference."
        },
    )
    coconut_prefix_text: str = field(
        default="<think><bot>",
        metadata={
            "help": "Assistant prefix appended before latent loop."
        },
    )
    coconut_forced_suffix_text: str = field(
        default="<eot></think><answer>",
        metadata={
            "help": "Suffix forcibly appended right after latent loop."
        },
    )
    debug_log_tokens: bool = field(
        default=False,
        metadata={
            "help": "Enable detailed token generation logging (may slow down evaluation)."
        },
    )
    use_coconut_latent_moe: bool = field(
        default=False,
        metadata={
            "help": "Enable latent MoE transition during COCONUT latent reasoning inference."
        },
    )
    coconut_latent_moe_num_experts: int = field(
        default=4,
        metadata={"help": "Number of routed experts in latent MoE transition."},
    )
    coconut_latent_moe_top_k: int = field(
        default=2,
        metadata={"help": "Top-k experts selected by MoE router each latent step."},
    )
    coconut_latent_moe_use_shared_expert: bool = field(
        default=True,
        metadata={"help": "Whether to include shared expert branch in latent MoE transition."},
    )
    coconut_latent_moe_step_embed_max_steps: int = field(
        default=32,
        metadata={"help": "Max latent steps covered by MoE step embedding table."},
    )
    coconut_latent_moe_context_type: str = field(
        default="prefix_last",
        metadata={
            "help": "Router context source for latent MoE. One of: none, prefix_last, disc."
        },
    )
    coconut_latent_moe_exclude_experts: str = field(
        default="",
        metadata={
            "help": "Eval-only: comma-separated routed expert indices to forbid in top-k (e.g. '2' or '0,3'). "
            "Router softmax is masked so these experts are never selected; shared expert unchanged."
        },
    )

def main():
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=720))
    if torch.cuda.is_available():
        local_rank_env = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank_env)
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # DEBUG PRINTS for Distributed Setup
    print_master("Distributed init debug info:")
    print_master(f"RANK: {os.environ.get('RANK')}")
    print_master(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print_master(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print_master(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print_master(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    if dist.is_initialized():
        print_rank(f"dist.get_rank(): {dist.get_rank()}")
        print_rank(f"dist.get_world_size(): {dist.get_world_size()}")

    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, EvalArguments))
    # Parse arguments
    model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    eval_args: EvalArguments
    if eval_args.use_coconut_latent_reasoning and training_args.per_device_eval_batch_size != 1:
        raise ValueError(
            "COCONUT latent reasoning eval currently requires "
            "--per_device_eval_batch_size 1."
        )
    if eval_args.use_coconut_latent_moe and (not eval_args.use_coconut_latent_reasoning):
        raise ValueError(
            "--use_coconut_latent_moe=True requires --use_coconut_latent_reasoning=True."
        )
    context_type = str(eval_args.coconut_latent_moe_context_type or "prefix_last").strip().lower()
    if context_type not in {"none", "prefix_last", "disc"}:
        raise ValueError(
            "--coconut_latent_moe_context_type must be one of: none, prefix_last, disc."
        )
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    # --- Model Loading ---
    base_model_path, adapter_path = _resolve_model_paths(model_args)
    if not getattr(model_args, "model_base", None):
        setattr(model_args, "model_base", base_model_path)
    hf_config = AutoConfig.from_pretrained(base_model_path)
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config, model_args.model_type)
    else:
        model_backbone = MODEL2BACKBONE.get(model_args.model_backbone, model_args.model_backbone)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_master(f"Model Backbone: {model_args.model_backbone}")
    print_master(f"[EVAL] base_model_path={base_model_path}")
    if adapter_path is not None:
        print_master(f"[EVAL] adapter_path={adapter_path}")

    # --- Model Loading (all ranks load in parallel from local disk) ---
    print_rank(f"[rank={local_rank}] Loading processor and model...")
    processor = load_processor(model_args, data_args)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    model = _load_qwen_generation_model(model_args, processor, eval_args=eval_args)
    print_rank(f"[rank={local_rank}] Model loaded.")

    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)
    print_rank(f"[rank={local_rank}] Model moved to device and ready.")

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)
    dataset_items = list(dataset_configs.items())
    if eval_args.dataset_names.strip():
        selected = [name.strip() for name in eval_args.dataset_names.split(",") if name.strip()]
        lower_to_name = {name.lower(): name for name in dataset_configs.keys()}
        filtered_items = []
        missing = []
        for raw_name in selected:
            matched_name = lower_to_name.get(raw_name.lower(), None)
            if matched_name is None:
                missing.append(raw_name)
                continue
            filtered_items.append((matched_name, dataset_configs[matched_name]))
        if missing:
            print_master(f"[EVAL] Warning: dataset(s) not found in yaml and will be skipped: {missing}")
        if not filtered_items:
            raise ValueError(
                f"No valid datasets selected from --dataset_names={eval_args.dataset_names}. "
                f"Available keys: {list(dataset_configs.keys())[:20]}..."
            )
        dataset_items = filtered_items
        print_master(f"[EVAL] Filtered datasets: {[name for name, _ in dataset_items]}")
    else:
        print_master(f"[EVAL] Evaluating all datasets in config ({len(dataset_items)} tasks).")


    # --- Main Evaluation Loop ---
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_items):
        # 0. load dataset
        if dist.is_initialized():
            _safe_dist_barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_{eval_args.qry_mode}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_{eval_args.tgt_mode}_tgt")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_{eval_args.qry_mode}_{eval_args.tgt_mode}_info.jsonl")
        query_meta_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_{eval_args.qry_mode}_qry_meta.json")
        cand_meta_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_{eval_args.tgt_mode}_tgt_meta.json")

        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)
        query_hit_max_new_tokens = None
        cand_hit_max_new_tokens = None

        if do_query or do_cand:
            print_master(f"[EVAL] Building datasets for task={dataset_name} ...")
            if data_args.data_basedir is not None:
                # Construct full paths for data files if --data_basedir is provided
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if data_args.data_basedir and task_config.get(key):
                        task_config[key] = _resolve_task_data_path(
                            data_args.data_basedir, task_config[key], key
                        )
            frame_root = task_config.get("frame_root", None)
            if frame_root:
                if (not os.path.isdir(frame_root)) or (next(os.scandir(frame_root), None) is None):
                    raise FileNotFoundError(
                        f"[EVAL] frame_root is missing or empty for task={dataset_name}: {frame_root}. "
                        "Video evaluation requires extracted frame directories. "
                        "Please extract archives under <data_basedir>/video-tasks/frames first "
                        "(e.g., video_cls.tar.gz / video_ret.tar.gz)."
                    )

            full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(model_args=model_args, data_args=data_args, **task_config)
            full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
            print_master(
                f"[EVAL] Dataset ready for task={dataset_name}: "
                f"queries={len(full_eval_qry_dataset)}, candidates={len(full_eval_cand_dataset)}"
            )
            eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
            # Pad datasets to be divisible by world_size before splitting
            if dist.is_initialized():
                padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)
                padded_cand_dataset, _ = pad_dataset_to_divisible(full_eval_cand_dataset, world_size)
                eval_qry_dataset = split_dataset_by_node(padded_qry_dataset, rank=local_rank, world_size=world_size)
                eval_cand_dataset = split_dataset_by_node(padded_cand_dataset, rank=local_rank, world_size=world_size)
            else:
                padded_qry_dataset, padded_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset

        if dist.is_initialized():
            _safe_dist_barrier()
        # --- 1. Compute Query Embeddings ---
        if do_query:
            print_master("Encoding queries...")
            eval_qry_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "qry")
            eval_qry_loader = DataLoader(eval_qry_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=eval_qry_collator, num_workers=training_args.dataloader_num_workers)
            query_embeds, gt_infos, query_hit_max_new_tokens = encode_embeddings(
                model,
                eval_qry_loader,
                training_args,
                model_args,
                padded_qry_dataset,
                encode_side="qry",
                description=f"Queries for {dataset_name}",
                processor=processor,
                mode=eval_args.qry_mode,
                eval_args=eval_args,
                valid_num_rows=len(full_eval_qry_dataset),
            )
            query_embeds = query_embeds[:len(full_eval_qry_dataset)]  # world_size>1, trim the padded data points
            gt_infos = gt_infos[:len(full_eval_qry_dataset)]
            if local_rank == 0:
                with open(query_embed_path, 'wb') as f:
                    pickle.dump(query_embeds, f)
                with open(dataset_info_path, 'w') as f:
                    for info in gt_infos:
                        f.write(json.dumps(info) + '\n')
                print_master(f"Saved query embeddings to {query_embed_path}")
                if eval_args.qry_mode == "gen" and eval_args.use_coconut_latent_reasoning:
                    with open(query_meta_path, "w") as f:
                        json.dump(
                            {
                                "hit_max_new_tokens": int(query_hit_max_new_tokens),
                                "num_samples": int(len(full_eval_qry_dataset)),
                            },
                            f,
                            indent=4,
                        )
                    print_master(
                        f"[EVAL][{dataset_name}] query samples hit max_new_tokens: "
                        f"{query_hit_max_new_tokens}/{len(full_eval_qry_dataset)}"
                    )
        if dist.is_initialized():
            _safe_dist_barrier()


        # --- 2. Compute Candidate Embeddings ---
        if do_cand:
            print_master("Encoding candidates...")
            eval_cand_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "cand")
            eval_cand_loader = DataLoader(eval_cand_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=eval_cand_collator, num_workers=training_args.dataloader_num_workers)

            cand_embeds, all_cand_ids, cand_hit_max_new_tokens = encode_embeddings(
                model,
                eval_cand_loader,
                training_args,
                model_args,
                padded_cand_dataset,
                encode_side="cand",
                description=f"Candidates for {dataset_name}",
                processor=processor,
                mode=eval_args.tgt_mode,
                eval_args=eval_args,
                valid_num_rows=len(full_eval_cand_dataset),
            )
            cand_embeds = cand_embeds[:len(full_eval_cand_dataset)]  # world_size>1, trim the padded data points
            all_cand_ids = all_cand_ids[:len(full_eval_cand_dataset)]

            if local_rank == 0:
                cand_embed_dict = {cand_id: embed for cand_id, embed in zip(all_cand_ids, cand_embeds)}
                with open(cand_embed_path, 'wb') as f: pickle.dump(cand_embed_dict, f)
                print_master(f"Saved candidate embeddings to {cand_embed_path}")
                if eval_args.tgt_mode == "gen" and eval_args.use_coconut_latent_reasoning:
                    with open(cand_meta_path, "w") as f:
                        json.dump(
                            {
                                "hit_max_new_tokens": int(cand_hit_max_new_tokens),
                                "num_samples": int(len(full_eval_cand_dataset)),
                            },
                            f,
                            indent=4,
                        )
                    print_master(
                        f"[EVAL][{dataset_name}] candidate samples hit max_new_tokens: "
                        f"{cand_hit_max_new_tokens}/{len(full_eval_cand_dataset)}"
                    )

        if dist.is_initialized():
            _safe_dist_barrier()

        # --- 3. Compute Scores (on master rank only) ---
        if local_rank == 0:
            if query_hit_max_new_tokens is None and os.path.exists(query_meta_path):
                try:
                    with open(query_meta_path, "r") as f:
                        query_hit_max_new_tokens = int(json.load(f).get("hit_max_new_tokens", 0))
                except Exception:
                    query_hit_max_new_tokens = None
            if cand_hit_max_new_tokens is None and os.path.exists(cand_meta_path):
                try:
                    with open(cand_meta_path, "r") as f:
                        cand_hit_max_new_tokens = int(json.load(f).get("hit_max_new_tokens", 0))
                except Exception:
                    cand_hit_max_new_tokens = None

            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_{eval_args.qry_mode}_{eval_args.tgt_mode}_score.json")
            if os.path.exists(score_path):
                try:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    print_master(f"Score of {dataset_name} (loaded from previous run): {score_path}")
                    formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
                    print_master(formatted)
                    continue
                except Exception as e:
                    print_master(f"Failed to load score for {dataset_name}, skipping {dataset_name}")
            with open(query_embed_path, 'rb') as f: qry_embeds = pickle.load(f)
            with open(cand_embed_path, 'rb') as f: cand_embed_dict = pickle.load(f)
            gt_infos = [json.loads(l) for l in open(dataset_info_path)]
            pred_dicts = []

            rank_against_all_candidates = task_config.get("eval_type", "global") == "global"
            if rank_against_all_candidates:
                cand_keys = list(cand_embed_dict.keys())
                cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])
                # Handle late-interaction scoring
                if qry_embeds.ndim == 3: # Query: [N_q, L_q, H] | Candidate: [N_c, L_c, H]
                    qry_embed = torch.from_numpy(qry_embeds)
                    cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]
                    scores = processor.score(qry_embed, cand_embeds, batch_size=64)  # use ColPali score function
                    ranked_candids = torch.argsort(-scores, dim=1).cpu().numpy().tolist()
                else: # Dense
                    cosine_scores = np.dot(qry_embeds, cand_embeds.T)
                    ranked_candids = np.argsort(-cosine_scores, axis=1)
                for qid, (ranked_candid, gt_info) in tqdm(enumerate(zip(ranked_candids, gt_infos)), desc=f"Calculating scores for {dataset_name}"):
                    rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                    rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None
                    assert rel_scores is None or len(rel_docids) == len(rel_scores)
                    pred_dicts.append({
                        "prediction": [cand_keys[i] for i in ranked_candid],
                        "label": rel_docids,
                        "rel_scores": rel_scores,
                    })
            else:
                for qid, (qry_embed, gt_info) in tqdm(enumerate(zip(qry_embeds, gt_infos)), desc=f"Calculating scores for {dataset_name}"):
                    cand_embeds = np.stack([cand_embed_dict[key] for key in gt_info["cand_names"]])
                    if qry_embeds.ndim == 3: # Query: [N_q, L_q, H] | Candidate: [N_c, L_c, H]
                        qry_embed = torch.from_numpy(np.array(qry_embed)).unsqueeze(0)
                        cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]
                        scores = processor.score(qry_embed, cand_embeds, batch_size=1024)  # use ColPali score function
                        ranked_candids = torch.argsort(-scores, dim=1).cpu().numpy().tolist()[0]
                    else:
                        cosine_score = np.dot(qry_embed, cand_embeds.T)
                        ranked_candids = np.argsort(-cosine_score)
                    rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                    rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None

                    assert rel_scores is None or len(rel_docids) == len(rel_scores)
                    pred_dicts.append({
                        "prediction": [gt_info["cand_names"][i] for i in ranked_candids],
                        "label": rel_docids,
                        "rel_scores": rel_scores,
                    })

            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_{eval_args.qry_mode}_{eval_args.tgt_mode}_score.json")
            pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_{eval_args.qry_mode}_{eval_args.tgt_mode}_pred.jsonl")

            metrics_to_report = task_config["metrics"] if task_config.get("metrics", None) is not None else ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
            metrics = RankingMetrics(metrics_to_report)
            score_dict = metrics.evaluate(pred_dicts)
            formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
            score_dict["num_pred"] = len(pred_dicts)
            score_dict["num_data"] = len(gt_infos)
            if eval_args.use_coconut_latent_reasoning and eval_args.qry_mode == "gen":
                qry_hit_count = int(query_hit_max_new_tokens) if query_hit_max_new_tokens is not None else 0
                score_dict["qry_hit_max_new_tokens"] = qry_hit_count
                score_dict["qry_hit_max_new_tokens_rate"] = (
                    f"{qry_hit_count / len(gt_infos) * 100:.2f}%" if len(gt_infos) > 0 else "0.00%"
                )
            if eval_args.use_coconut_latent_reasoning and eval_args.tgt_mode == "gen":
                tgt_hit_count = int(cand_hit_max_new_tokens) if cand_hit_max_new_tokens is not None else 0
                cand_total = len(cand_embed_dict) if cand_embed_dict is not None else 0
                score_dict["tgt_hit_max_new_tokens"] = tgt_hit_count
                score_dict["tgt_hit_max_new_tokens_rate"] = (
                    f"{tgt_hit_count / cand_total * 100:.2f}%" if cand_total > 0 else "0.00%"
                )
            print_master(f"Score of {dataset_name}:")
            print_master(formatted)
            print_master(f"Outputting final score to: {score_path}")
            with open(score_path, "w") as f:
                json.dump(score_dict, f, indent=4)
            with open(pred_path, "w") as f:
                for pred in pred_dicts:
                    f.write(json.dumps(pred) + '\n')


if __name__ == "__main__":
    main()
