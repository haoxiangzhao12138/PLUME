#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Multi-node launcher for PLUME latent-reasoning training (8 nodes × 8 GPUs)
#
# Usage:
#   On EACH node, run this script with the correct NODE_RANK:
#     NODE_RANK=0 bash scripts/train_multinode.sh   # on 113 (master)
#     NODE_RANK=1 bash scripts/train_multinode.sh   # on 114
#     ...
#     NODE_RANK=7 bash scripts/train_multinode.sh   # on 127
#
#   Or use launch_multinode.sh to launch all 8 nodes from 113 via SSH.
# ============================================================================
# ---------- Cluster topology ----------
MASTER_ADDR="${MASTER_ADDR:-192.168.100.113}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# ---------- Paths (GPFS shared) ----------
WORK_DIR="${WORK_DIR:-.}"
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/path/to/annotations.jsonl}"
MEDIA_ROOT="${MEDIA_ROOT:-/path/to/media_root}"
# SUBSET_FILTER="${SUBSET_FILTER:-CIRR,MSCOCO_t2i,WebQA,ImageNet-1K,RefCOCO,InfographicsVQA}"  # empty = ALL datasets
SUBSET_FILTER="${SUBSET_FILTER:-}"
# ---------- Training hyperparams ----------
# 8 nodes × 8 GPUs = 64 GPUs
# effective_global_batch = 2 * 4 * 64 = 512
# contrastive_batch (per step) = 2 * 64 = 128
PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-2}"
MAX_LEN="${MAX_LEN:-12288}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
# Visual resolution controls (same semantics as DataArguments)
MAX_PIXELS="${MAX_PIXELS:-2359296}"                  # 28*28*576
MIN_PIXELS="${MIN_PIXELS:-768}"                  # 28*28*16 (match original)
VIDEO_MAX_FRAME_PIXELS="${VIDEO_MAX_FRAME_PIXELS:-2359296}"  # 32*28*28
VIDEO_MIN_FRAME_PIXELS="${VIDEO_MIN_FRAME_PIXELS:-768}"   # 4*28*28

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj}"
LATENT_MOE_ENABLE="${LATENT_MOE_ENABLE:-True}"
LATENT_MOE_NUM_EXPERTS="${LATENT_MOE_NUM_EXPERTS:-4}"
LATENT_MOE_TOP_K="${LATENT_MOE_TOP_K:-2}"
LATENT_MOE_USE_SHARED_EXPERT="${LATENT_MOE_USE_SHARED_EXPERT:-True}"
LATENT_MOE_BALANCE_LOSS_WEIGHT="${LATENT_MOE_BALANCE_LOSS_WEIGHT:-0.1}"
LATENT_MOE_STEP_EMBED_MAX_STEPS="${LATENT_MOE_STEP_EMBED_MAX_STEPS:-32}"
LATENT_MOE_CONTEXT_TYPE="${LATENT_MOE_CONTEXT_TYPE:-disc}"
LATENT_MOE_EXPERT_DROPOUT="${LATENT_MOE_EXPERT_DROPOUT:-0.1}"

THINK_SEGMENTS="${THINK_SEGMENTS:-6}"
CT_PER_SEG="${CT_PER_SEG:-1}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-subset_balanced}"
FINAL_STAGE_PORTION="${FINAL_STAGE_PORTION:-0.5}"
LATENT_ANSWER_IN_FINAL_HALF="${LATENT_ANSWER_IN_FINAL_HALF:-True}"

FINAL_STAGE_ANSWER_PORTION="${FINAL_STAGE_ANSWER_PORTION:-0.5}"

GEN_CONTRASTIVE_W="${GEN_CONTRASTIVE_W:-1.0}"
DISC_CONTRASTIVE_W="${DISC_CONTRASTIVE_W:-1.0}"
CONTRASTIVE_LOGIT_SCALE="${CONTRASTIVE_LOGIT_SCALE:-50.0}"

SAVE_STEPS="${SAVE_STEPS:-100}"
LOG_STEPS="${LOG_STEPS:-10}"
WANDB_MODE="${WANDB_MODE:-disabled}"

OUTPUT_DIR="${OUTPUT_DIR:-output/test/PLUME-multinode$(date +%Y-%m-%d-%H-%M-%S)}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-${WORK_DIR}/configs/deepspeed/zero3.json}"

# ---------- Communication profile ----------
# Profiles:
#   tcp_stable (default): current safe setup over TCP/bond0
#   tcp_fast           : more aggressive TCP channels/threads
#   rdma_fast          : prefer IB/RDMA (if fabric is healthy)
COMM_PROFILE="${COMM_PROFILE:-tcp_stable}"
COMM_PROFILE_LC="$(echo "${COMM_PROFILE}" | tr '[:upper:]' '[:lower:]')"

detect_active_ib_hcas() {
    local out=()
    local dev state link
    for dev_path in /sys/class/infiniband/*; do
        [ -d "${dev_path}" ] || continue
        dev="$(basename "${dev_path}")"
        state="$(cat "${dev_path}/ports/1/state" 2>/dev/null || true)"
        link="$(cat "${dev_path}/ports/1/link_layer" 2>/dev/null || true)"
        if [[ "${state}" == *"ACTIVE"* && "${link}" == "InfiniBand" ]]; then
            out+=("${dev}")
        fi
    done
    if [ "${#out[@]}" -gt 0 ]; then
        local IFS=,
        echo "${out[*]}"
    fi
}

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
# H20 + NCCL 2.27 may occasionally hang on NVLS/CUMEM paths in large multi-node setups.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"       # 8MB buffer, fewer small packets
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

case "${COMM_PROFILE_LC}" in
    rdma_fast)
        # Use IB/RDMA for collectives; keep socket iface for bootstrap/Gloo.
        export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
        export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
        export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
        export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
        export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
        export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
        export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
        if [ -z "${NCCL_IB_HCA:-}" ]; then
            DETECTED_IB_HCA="$(detect_active_ib_hcas || true)"
            if [ -n "${DETECTED_IB_HCA}" ]; then
                export NCCL_IB_HCA="${DETECTED_IB_HCA}"
            fi
        fi
        ;;
    tcp_fast)
        export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
        export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
        export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-2}"
        export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
        export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-8}"
        export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
        ;;
    tcp_stable|*)
        export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
        export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
        export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-1}"
        export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
        export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
        export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
        ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# TCP keepalive: detect dead peers faster instead of hanging forever
sysctl -w net.ipv4.tcp_keepalive_time=60  2>/dev/null || true
sysctl -w net.ipv4.tcp_keepalive_intvl=10 2>/dev/null || true
sysctl -w net.ipv4.tcp_keepalive_probes=5 2>/dev/null || true

# ---------- Print info ----------
TOTAL_GPUS=$(( NNODES * NPROC_PER_NODE ))
GLOBAL_BATCH=$(( PER_DEVICE_BS * GRAD_ACC * TOTAL_GPUS ))
CONTRASTIVE_BATCH=$(( PER_DEVICE_BS * TOTAL_GPUS ))
echo "============================================="
echo "[PLUME-MULTINODE] node_rank=${NODE_RANK}/${NNODES}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[PLUME-MULTINODE] total_gpus=${TOTAL_GPUS}, per_device_bs=${PER_DEVICE_BS}, grad_acc=${GRAD_ACC}"
echo "[PLUME-MULTINODE] effective_global_batch=${GLOBAL_BATCH}, contrastive_batch=${CONTRASTIVE_BATCH}"
echo "[PLUME-MULTINODE] pixels image=${MIN_PIXELS}~${MAX_PIXELS}, video_frame=${VIDEO_MIN_FRAME_PIXELS}~${VIDEO_MAX_FRAME_PIXELS}"
echo "[PLUME-MULTINODE] latent_moe enable=${LATENT_MOE_ENABLE}, experts=${LATENT_MOE_NUM_EXPERTS}, top_k=${LATENT_MOE_TOP_K}, ctx=${LATENT_MOE_CONTEXT_TYPE}, balance_w=${LATENT_MOE_BALANCE_LOSS_WEIGHT}, expert_dropout=${LATENT_MOE_EXPERT_DROPOUT}"
echo "[PLUME-MULTINODE] deepspeed use=${USE_DEEPSPEED}, cfg=${DEEPSPEED_CFG}"
echo "[PLUME-MULTINODE] comm_profile=${COMM_PROFILE_LC}, NCCL_IB_DISABLE=${NCCL_IB_DISABLE}, NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}, NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS}, NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS}"
echo "[PLUME-MULTINODE] NCCL_IB_HCA=${NCCL_IB_HCA:-<auto-or-unset>}"
echo "[PLUME-MULTINODE] output_dir=${OUTPUT_DIR}"
echo "============================================="

cd "${WORK_DIR}"

# Activate conda env (needed for non-interactive SSH sessions)
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate plume
fi

DEEPSPEED_ARGS=()
case "${USE_DEEPSPEED,,}" in
    1|true|yes|on)
        DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CFG}")
        ;;
esac

WANDB_MODE="${WANDB_MODE}" \
"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
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
  --max_pixels "${MAX_PIXELS}" \
  --min_pixels "${MIN_PIXELS}" \
  --video_max_frame_pixels "${VIDEO_MAX_FRAME_PIXELS}" \
  --video_min_frame_pixels "${VIDEO_MIN_FRAME_PIXELS}" \
  --gradient_checkpointing True \
  --max_grad_norm 1 \
  --dataloader_num_workers 0 \
  --save_total_limit 20 \
  --use_lora False \
  --latent_moe_enable "${LATENT_MOE_ENABLE}" \
  --latent_moe_num_experts "${LATENT_MOE_NUM_EXPERTS}" \
  --latent_moe_top_k "${LATENT_MOE_TOP_K}" \
  --latent_moe_use_shared_expert "${LATENT_MOE_USE_SHARED_EXPERT}" \
  --latent_moe_balance_loss_weight "${LATENT_MOE_BALANCE_LOSS_WEIGHT}" \
  --latent_moe_step_embed_max_steps "${LATENT_MOE_STEP_EMBED_MAX_STEPS}" \
  --latent_moe_context_type "${LATENT_MOE_CONTEXT_TYPE}" \
  --latent_moe_expert_dropout "${LATENT_MOE_EXPERT_DROPOUT}" \
  --tune_mm_llm True \
  --tune_mm_mlp True \
  --tune_mm_vision False \
  --plume_annotation_path "${ANNOTATION_PATH}" \
  --plume_subset_filter "${SUBSET_FILTER}" \
  --plume_media_root "${MEDIA_ROOT}" \
  --plume_use_qry True \
  --plume_use_pos True \
  --plume_sampling_strategy "${SAMPLING_STRATEGY}" \
  --plume_curriculum_stages 0.25,0.5,0.75,1.0 \
  --plume_final_stage_portion "${FINAL_STAGE_PORTION}" \
  --plume_latent_answer_in_final_half "${LATENT_ANSWER_IN_FINAL_HALF}" \
  --plume_final_stage_answer_portion "${FINAL_STAGE_ANSWER_PORTION}" \
  --plume_think_segments "${THINK_SEGMENTS}" \
  --plume_ct_tokens_per_segment "${CT_PER_SEG}" \
  --plume_include_gen_emb_loss True \
  --plume_gen_contrastive_weight "${GEN_CONTRASTIVE_W}" \
  --plume_disc_contrastive_weight "${DISC_CONTRASTIVE_W}" \
  --plume_contrastive_logit_scale "${CONTRASTIVE_LOGIT_SCALE}" \
  --plume_contrastive_cross_device True \
  --plume_contrastive_local_loss True \
  --plume_oom_precheck_batches 1 \
  --plume_enable_oom_precheck False \
  --plume_oom_precheck_subsets "K700,Video-MME,YouCook2" \
  --plume_oom_precheck_batches 2 \
  --ddp_timeout 3600 \
  ${RESUME_CKPT:+--resume_from_checkpoint "$RESUME_CKPT"}
# for node in 192.168.100.113 192.168.100.24 192.168.100.28 192.168.100.33 192.168.100.34 192.168.100.115 192.168.100.116 192.168.100.127; do
#   echo "Killing on $node..."
#   ssh "$node" "pkill -f train_plume_gc.py; pkill -f torchrun" 2>/dev/null &
# done