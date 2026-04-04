"""
Gradient-checkpointing friendly PLUME training entry.

This script keeps the original latent reasoning path:
  1) prefix -> cache
  2) latent loop with previous hidden as inputs_embeds
  3) suffix teacher forcing with masked CE

Difference from train_plume.py:
  - `--gradient_checkpointing True` enables a manual activation-checkpoint wrapper
    around each single-sample latent forward, instead of enabling HF built-in
    model gradient checkpointing (which conflicts with `use_cache=True`).

Example:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  plume/train/train_plume_gc.py \
  --model_name_or_path /path/to/model \
  --output_dir output/PLUME-gc-$(date +%Y-%m-%d-%H-%M-%S) \
  --bf16 \
  --gradient_checkpointing True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --use_lora True
"""

import json
import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint_utils
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2VLImageProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import plume.train.train_plume as plume_base
from plume.data.data_plume import make_plume_data_module
from plume.train.argument import DataArguments, ModelArguments, TrainingArguments

local_rank = None


class PlumeGradientCheckpointTrainer(plume_base.PlumeTrainer):
    def __init__(
        self,
        *args,
        enable_manual_gradient_checkpointing: bool = False,
        manual_gc_use_reentrant: bool = False,
        **kwargs,
    ):
        self.enable_manual_gradient_checkpointing = bool(enable_manual_gradient_checkpointing)
        self.manual_gc_use_reentrant = bool(manual_gc_use_reentrant)
        super().__init__(*args, **kwargs)

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
    ):
        if (not self.enable_manual_gradient_checkpointing) or (not model.training):
            return plume_base.PlumeTrainer._single_sample_loss(
                self,
                model=model,
                prefix_input_ids=prefix_input_ids,
                prefix_attention_mask=prefix_attention_mask,
                prefix_position_ids=prefix_position_ids,
                suffix_input_ids=suffix_input_ids,
                suffix_attention_mask=suffix_attention_mask,
                suffix_position_ids=suffix_position_ids,
                suffix_labels=suffix_labels,
                latent_steps=latent_steps,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        hidden_size, rep_dtype = self._get_rep_meta_cached(model)

        def _forward_fn(
            prefix_input_ids_t,
            prefix_attention_mask_t,
            prefix_position_ids_t,
            suffix_input_ids_t,
            suffix_attention_mask_t,
            suffix_position_ids_t,
            suffix_labels_t,
        ):
            output = plume_base.PlumeTrainer._single_sample_loss(
                self,
                model=model,
                prefix_input_ids=prefix_input_ids_t,
                prefix_attention_mask=prefix_attention_mask_t,
                prefix_position_ids=prefix_position_ids_t,
                suffix_input_ids=suffix_input_ids_t,
                suffix_attention_mask=suffix_attention_mask_t,
                suffix_position_ids=suffix_position_ids_t,
                suffix_labels=suffix_labels_t,
                latent_steps=latent_steps,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

            if output is None:
                zero_loss = prefix_input_ids_t.sum() * 0.0
                device = zero_loss.device
                zero_scalar = torch.zeros((), device=device, dtype=zero_loss.dtype)
                false_flag = torch.zeros((), device=device, dtype=torch.bool)
                zero_rep = torch.zeros((hidden_size,), device=device, dtype=rep_dtype)
                return (
                    zero_loss,
                    false_flag,
                    zero_scalar,
                    zero_scalar,
                    zero_rep,
                    false_flag,
                    zero_rep,
                    false_flag,
                )

            sample_loss, sample_latent_steps, sample_suffix_tokens, gen_rep, disc_rep = output
            device = sample_loss.device
            valid_flag = torch.ones((), device=device, dtype=torch.bool)
            latent_steps_tensor = sample_loss.new_tensor(float(sample_latent_steps))
            suffix_tokens_tensor = sample_loss.new_tensor(float(sample_suffix_tokens))

            if gen_rep is None:
                gen_rep_tensor = torch.zeros((hidden_size,), device=device, dtype=rep_dtype)
                gen_valid_flag = torch.zeros((), device=device, dtype=torch.bool)
            else:
                gen_rep_tensor = gen_rep.to(device=device, dtype=rep_dtype)
                gen_valid_flag = torch.ones((), device=device, dtype=torch.bool)

            if disc_rep is None:
                disc_rep_tensor = torch.zeros((hidden_size,), device=device, dtype=rep_dtype)
                disc_valid_flag = torch.zeros((), device=device, dtype=torch.bool)
            else:
                disc_rep_tensor = disc_rep.to(device=device, dtype=rep_dtype)
                disc_valid_flag = torch.ones((), device=device, dtype=torch.bool)

            return (
                sample_loss,
                valid_flag,
                latent_steps_tensor,
                suffix_tokens_tensor,
                gen_rep_tensor,
                gen_valid_flag,
                disc_rep_tensor,
                disc_valid_flag,
            )

        (
            sample_loss,
            sample_valid_flag,
            latent_steps_tensor,
            suffix_tokens_tensor,
            gen_rep_tensor,
            gen_valid_flag,
            disc_rep_tensor,
            disc_valid_flag,
        ) = checkpoint_utils.checkpoint(
            _forward_fn,
            prefix_input_ids,
            prefix_attention_mask,
            prefix_position_ids,
            suffix_input_ids,
            suffix_attention_mask,
            suffix_position_ids,
            suffix_labels,
            use_reentrant=self.manual_gc_use_reentrant,
        )

        if not bool(sample_valid_flag.detach().item()):
            return None

        gen_rep = gen_rep_tensor if bool(gen_valid_flag.detach().item()) else None
        disc_rep = disc_rep_tensor if bool(disc_valid_flag.detach().item()) else None
        return (
            sample_loss,
            int(latent_steps_tensor.detach().item()),
            int(suffix_tokens_tensor.detach().item()),
            gen_rep,
            disc_rep,
        )


def train(attn_implementation: str = "flash_attention_2"):
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    training_args.data_group = data_args.data_group
    os.makedirs(training_args.output_dir, exist_ok=True)

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
        plume_base.rank0_print(f"[PLUME-GC] Failed to init log file {log_file}: {e}")

    training_args.remove_unused_columns = False

    manual_gc_enabled = bool(training_args.gradient_checkpointing)
    manual_gc_use_reentrant = False
    gc_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None)
    if isinstance(gc_kwargs, dict):
        manual_gc_use_reentrant = bool(gc_kwargs.get("use_reentrant", False))

    if manual_gc_enabled:
        plume_base.rank0_print(
            "[PLUME-GC] gradient_checkpointing=True detected. "
            "Use manual activation checkpointing and keep latent KV-cache logic unchanged."
        )
        if manual_gc_use_reentrant:
            plume_base.rank0_print("[PLUME-GC] manual checkpoint uses reentrant=True")
    else:
        plume_base.rank0_print("[PLUME-GC] gradient_checkpointing=False, manual checkpoint disabled")

    # Disable HF built-in gradient checkpointing to avoid conflict with use_cache=True.
    training_args.gradient_checkpointing = False
    training_args.gradient_checkpointing_kwargs = {}

    attn_implementation = plume_base.resolve_attn_implementation(model_args.attn_implementation)
    model, attn_implementation = plume_base.load_qwen_model_with_fallback(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    plume_base.rank0_print(f"[PLUME-GC] Using attn_implementation={attn_implementation}")

    image_processor_src = os.environ.get("IMAGE_PROCESSOR_PATH", "").strip() or model_args.model_name_or_path
    try:
        if "qwen2.5" in model_args.model_name_or_path.lower():
            data_args.image_processor = AutoProcessor.from_pretrained(
                image_processor_src
            ).image_processor
        else:
            data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
                image_processor_src
            )
    except OSError as e:
        fallback_src = ""
        config_path = os.path.join(str(model_args.model_name_or_path), "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    fallback_src = str(json.load(f).get("_name_or_path", "")).strip()
            except Exception:
                fallback_src = ""
        if not fallback_src:
            fallback_src = str(getattr(model.config, "_name_or_path", "")).strip()
        if fallback_src:
            plume_base.rank0_print(
                f"[PLUME-GC] image processor load failed from {image_processor_src}: {e}. "
                f"Fallback to base model path: {fallback_src}"
            )
            if "qwen2.5" in str(fallback_src).lower():
                data_args.image_processor = AutoProcessor.from_pretrained(
                    fallback_src
                ).image_processor
            else:
                data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
                    fallback_src
                )
        else:
            raise

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    plume_base.initialize_new_tokens(model, tokenizer, model_args.model_name_or_path,
                                       force_reinit_all=model_args.plume_force_reinit_all_tokens)
    plume_base.set_model(model_args, model)
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
        plume_base._unwrap_model(model).get_input_embeddings().requires_grad_(True)
        plume_base._get_lm_head(model).requires_grad_(True)

    plume_base.initialize_latent_moe(model, model_args)

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        plume_base.safe_print_trainable_parameters(model, use_lora=model_args.use_lora)

    data_module = make_plume_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        enable_lazy_tokenization=(training_args.dataloader_num_workers > 0),
    )
    stage_callback = plume_base.CurriculumStageCallback(data_module["train_dataset"])
    train_dataset = data_module["train_dataset"]
    global_batch = (
        int(training_args.per_device_train_batch_size)
        * max(1, int(training_args.gradient_accumulation_steps))
        * max(1, int(training_args.world_size))
    )
    plume_base.rank0_print("[PLUME-GC][TRAIN] ===== Training Setup =====")
    plume_base.rank0_print(f"[PLUME-GC][TRAIN] train_samples={len(train_dataset)}")
    plume_base.rank0_print(
        f"[PLUME-GC][TRAIN] per_device_batch={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, world_size={training_args.world_size}, "
        f"effective_global_batch={global_batch}"
    )
    plume_base.rank0_print(
        f"[PLUME-GC][TRAIN] max_steps={training_args.max_steps}, "
        f"num_train_epochs={training_args.num_train_epochs}, "
        f"learning_rate={training_args.learning_rate}"
    )
    plume_base.rank0_print(
        f"[PLUME-GC][TRAIN] contrastive_weights(gen/disc)="
        f"{data_args.plume_gen_contrastive_weight}/{data_args.plume_disc_contrastive_weight}, "
        f"logit_scale={data_args.plume_contrastive_logit_scale}, "
        f"debug_disc_oracle_pos_from_qry={data_args.plume_debug_disc_oracle_pos_from_qry}"
    )
    plume_base.rank0_print(
        f"[PLUME-GC][TRAIN] sampling_strategy={data_args.plume_sampling_strategy}, "
        f"data_group={data_args.data_group}, "
        f"final_stage_portion={data_args.plume_final_stage_portion}, "
        f"latent_answer_in_final_half={data_args.plume_latent_answer_in_final_half}, "
        f"final_stage_answer_portion={data_args.plume_final_stage_answer_portion}"
    )
    plume_base.rank0_print(
        f"[PLUME-GC][TRAIN] lora: enabled={model_args.use_lora}, r={model_args.lora_r}, "
        f"alpha={model_args.lora_alpha}, dropout={model_args.lora_dropout}, "
        f"use_dora={model_args.lora_use_dora}"
    )
    plume_base.rank0_print(
        f"[PLUME-GC][TRAIN] deepspeed_enabled={plume_base._is_deepspeed_enabled(training_args)}, "
        f"deepspeed_cfg={getattr(training_args, 'deepspeed', None)}"
    )
    plume_base.rank0_print(
        f"[PLUME-GC][TRAIN] latent_moe: enabled={model_args.latent_moe_enable}, "
        f"num_experts={model_args.latent_moe_num_experts}, top_k={model_args.latent_moe_top_k}, "
        f"use_shared_expert={model_args.latent_moe_use_shared_expert}, "
        f"context_type={model_args.latent_moe_context_type}, "
        f"balance_w={model_args.latent_moe_balance_loss_weight}"
    )
    plume_base.rank0_print(
        f"[PLUME-GC][TRAIN] manual_gradient_checkpointing={manual_gc_enabled}, "
        f"use_reentrant={manual_gc_use_reentrant}"
    )
    plume_base.rank0_print("[PLUME-GC][TRAIN] =========================")

    gen_emb_token_id = tokenizer.convert_tokens_to_ids("<gen_emb>")
    disc_emb_token_id = tokenizer.convert_tokens_to_ids("<disc_emb>")
    trainer = PlumeGradientCheckpointTrainer(
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
        enable_manual_gradient_checkpointing=manual_gc_enabled,
        manual_gc_use_reentrant=manual_gc_use_reentrant,
        callbacks=[stage_callback],
        **data_module,
    )
    stage_callback.set_trainer(trainer)
    plume_base._run_oom_precheck(trainer, train_dataset, data_args)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    model.config.use_cache = True
    plume_base.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
