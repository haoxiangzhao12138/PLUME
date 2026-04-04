#!/usr/bin/env bash
set -euo pipefail

# Large-batch launcher for latent-reasoning + manual gradient checkpointing.
# Override any variable inline, e.g.:
#   PER_DEVICE_BS=12 GRAD_ACC=8 bash scripts/train_singlenode.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
WORK_DIR="${WORK_DIR:-.}"

MODEL_PATH="${MODEL_PATH:-/path/to/model}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/path/to/annotations.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-/path/to/media_root}"
SUBSET_FILTER="${SUBSET_FILTER:-}"
MAX_PIXELS="${MAX_PIXELS:-2359296}"                  # 28*28*576
MIN_PIXELS="${MIN_PIXELS:-768}"     
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACC="${GRAD_ACC:-1}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1}"
MAX_LEN="${MAX_LEN:-11288}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj}"
LATENT_MOE_ENABLE="${LATENT_MOE_ENABLE:-True}"
LATENT_MOE_NUM_EXPERTS="${LATENT_MOE_NUM_EXPERTS:-4}"
LATENT_MOE_TOP_K="${LATENT_MOE_TOP_K:-2}"
LATENT_MOE_USE_SHARED_EXPERT="${LATENT_MOE_USE_SHARED_EXPERT:-True}"
LATENT_MOE_BALANCE_LOSS_WEIGHT="${LATENT_MOE_BALANCE_LOSS_WEIGHT:-0.1}"
LATENT_MOE_STEP_EMBED_MAX_STEPS="${LATENT_MOE_STEP_EMBED_MAX_STEPS:-32}"
LATENT_MOE_CONTEXT_TYPE="${LATENT_MOE_CONTEXT_TYPE:-none}"

FINAL_STAGE_PORTION="${FINAL_STAGE_PORTION:-0.5}"
LATENT_ANSWER_IN_FINAL_HALF="${LATENT_ANSWER_IN_FINAL_HALF:-True}"
FINAL_STAGE_ANSWER_PORTION="${FINAL_STAGE_ANSWER_PORTION:-0.5}"

THINK_SEGMENTS="${THINK_SEGMENTS:-4}"
CT_PER_SEG="${CT_PER_SEG:-1}"

GEN_CONTRASTIVE_W="${GEN_CONTRASTIVE_W:-1.0}"
DISC_CONTRASTIVE_W="${DISC_CONTRASTIVE_W:-1.0}"
CONTRASTIVE_LOGIT_SCALE="${CONTRASTIVE_LOGIT_SCALE:-50.0}"
DEBUG_DISC_ORACLE_POS_FROM_QRY="${DEBUG_DISC_ORACLE_POS_FROM_QRY:-False}"

SAVE_STEPS="${SAVE_STEPS:-50}"
LOG_STEPS="${LOG_STEPS:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
PLUME_ENABLE_OOM_PRECHECK="${PLUME_ENABLE_OOM_PRECHECK:-False}"
PLUME_OOM_PRECHECK_BATCHES="${PLUME_OOM_PRECHECK_BATCHES:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/output/test/PLUME-train-$(date +%Y-%m-%d-%H-%M-%S)}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-${WORK_DIR}/configs/deepspeed/zero3.json}"

# NCCL stability knobs (aligned with multinode launcher defaults).
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-1}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

GLOBAL_BATCH=$(( PER_DEVICE_BS * GRAD_ACC * NPROC_PER_NODE ))
echo "[PLUME-LAUNCH] nproc=${NPROC_PER_NODE}, per_device_bs=${PER_DEVICE_BS}, grad_acc=${GRAD_ACC}, effective_global_batch=${GLOBAL_BATCH}"
echo "[PLUME-LAUNCH] latent_moe enable=${LATENT_MOE_ENABLE}, experts=${LATENT_MOE_NUM_EXPERTS}, top_k=${LATENT_MOE_TOP_K}, ctx=${LATENT_MOE_CONTEXT_TYPE}, balance_w=${LATENT_MOE_BALANCE_LOSS_WEIGHT}"
echo "[PLUME-LAUNCH] deepspeed use=${USE_DEEPSPEED}, cfg=${DEEPSPEED_CFG}"
echo "[PLUME-LAUNCH] output_dir=${OUTPUT_DIR}"
echo "[PLUME-LAUNCH] work_dir=${WORK_DIR}"
echo "[PLUME-LAUNCH] pwd(before cd)=$(pwd)"

cd "${WORK_DIR}"
mkdir -p "${OUTPUT_DIR}"
echo "[PLUME-LAUNCH] pwd(after cd)=$(pwd)"

DEEPSPEED_ARGS=()
case "${USE_DEEPSPEED,,}" in
  1|true|yes|on)
    DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CFG}")
    ;;
esac

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" WANDB_MODE="${WANDB_MODE}" \
"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" \
  plume/train/train_plume_gc.py \
  "${DEEPSPEED_ARGS[@]}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --attn_implementation flash_attention_2 \
  --bf16 \
  --learning_rate "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --num_train_epochs "${EPOCHS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOG_STEPS}" \
  --model_max_length "${MAX_LEN}" \
  --gradient_checkpointing True \
  --use_lora False \
  --latent_moe_enable "${LATENT_MOE_ENABLE}" \
  --latent_moe_num_experts "${LATENT_MOE_NUM_EXPERTS}" \
  --latent_moe_top_k "${LATENT_MOE_TOP_K}" \
  --latent_moe_use_shared_expert "${LATENT_MOE_USE_SHARED_EXPERT}" \
  --latent_moe_balance_loss_weight "${LATENT_MOE_BALANCE_LOSS_WEIGHT}" \
  --latent_moe_step_embed_max_steps "${LATENT_MOE_STEP_EMBED_MAX_STEPS}" \
  --latent_moe_context_type "${LATENT_MOE_CONTEXT_TYPE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_use_dora False \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --tune_mm_llm True \
  --tune_mm_mlp True \
  --max_pixels "${MAX_PIXELS}" \
  --min_pixels "${MIN_PIXELS}" \
  --tune_mm_vision False \
  --plume_annotation_path "${ANNOTATION_PATH}" \
  --plume_subset_filter "${SUBSET_FILTER}" \
  --plume_media_root "${MEDIA_ROOT}" \
  --plume_use_qry True \
  --plume_use_pos True \
  --plume_curriculum_stages 1 \
  --plume_think_segments "${THINK_SEGMENTS}" \
  --plume_final_stage_portion "${FINAL_STAGE_PORTION}" \
  --plume_latent_answer_in_final_half "${LATENT_ANSWER_IN_FINAL_HALF}" \
  --plume_final_stage_answer_portion "${FINAL_STAGE_ANSWER_PORTION}" \
  --plume_ct_tokens_per_segment "${CT_PER_SEG}" \
  --plume_include_gen_emb_loss True \
  --plume_gen_contrastive_weight "${GEN_CONTRASTIVE_W}" \
  --plume_disc_contrastive_weight "${DISC_CONTRASTIVE_W}" \
  --plume_contrastive_logit_scale "${CONTRASTIVE_LOGIT_SCALE}" \
  --plume_contrastive_cross_device True \
  --plume_contrastive_local_loss True \
  --plume_debug_disc_oracle_pos_from_qry "${DEBUG_DISC_ORACLE_POS_FROM_QRY}" \
  --plume_oom_precheck_batches "${PLUME_OOM_PRECHECK_BATCHES}" \
  --plume_enable_oom_precheck "${PLUME_ENABLE_OOM_PRECHECK}"
  # --plume_force_reinit_all_tokens True


