#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Launch all 8 nodes from master (NODES[0])
# Usage: bash scripts/launch_multinode.sh
# Optional WANDB env:
#   WANDB_MODE=online|offline|disabled
#   WANDB_PROJECT=PLUME
#   WANDB_NAME=<run_name>
#   WANDB_RUN_GROUP=<group_name>
#   WANDB_ENTITY=<team_or_user>
#   WANDB_RUN_ID=<stable_id_for_resume>
#   WANDB_RESUME=allow|must|never
#   WANDB_TAGS=multinode,plume
# ============================================================================
# wandb启动：
# WANDB_MODE=online \
# WANDB_PROJECT=PLUME \
# WANDB_NAME=fewdata-4node-$(date +%m%d-%H%M) \
# WANDB_RUN_GROUP=plume-fewdata \
# bash scripts/launch_multinode.sh
# Clear stale env vars so train_multinode.sh uses its own defaults
unset PER_DEVICE_BS GRAD_ACC LR MAX_LEN NNODES WARMUP_RATIO LR_SCHEDULER WEIGHT_DECAY 2>/dev/null || true

NODES=("${PLUME_NODE_0:-localhost}" "${PLUME_NODE_1:-localhost}")
WORK_DIR="${WORK_DIR:-.}"
SCRIPT="scripts/train_multinode.sh"
CONDA_ENV="${CONDA_ENV:-plume}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
MASTER_ADDR="${MASTER_ADDR:-${NODES[0]}}"
MASTER_PORT="${MASTER_PORT:-29500}"
# 
# OUTPUT_DIR="${OUTPUT_DIR:-output/PLUME/PLUME-NoAns-6latent-8node-moe-2epoch$(date +%Y-%m-%d-%H-%M-%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-output/PLUME/PLUME-multinode-$(date +%Y-%m-%d-%H-%M-%S)}"
export OUTPUT_DIR
RUN_BASENAME="$(basename "${OUTPUT_DIR}")"

# Resume from checkpoint (optional)
RESUME_CKPT="${RESUME_CKPT:-}"
# RESUME_CKPT="${RESUME_CKPT:-}"
export RESUME_CKPT
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-${WORK_DIR}/configs/deepspeed/zero3.json}"
export USE_DEEPSPEED DEEPSPEED_CFG

# Weights & Biases settings
WANDB_MODE="${WANDB_MODE:-offline}"           # online | offline | disabled
WANDB_PROJECT="${WANDB_PROJECT:-PLUME}"
WANDB_NAME="${WANDB_NAME:-${RUN_BASENAME}}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_BASENAME}}"
WANDB_RESUME="${WANDB_RESUME:-never}"
WANDB_TAGS="${WANDB_TAGS:-multinode,plume}"

export WANDB_MODE WANDB_PROJECT WANDB_NAME WANDB_RUN_GROUP WANDB_RESUME WANDB_TAGS

# Communication profile passed to train_multinode.sh:
#   tcp_stable (default), tcp_fast, rdma_fast
COMM_PROFILE="${COMM_PROFILE:-tcp_stable}"
export COMM_PROFILE

# Latent MoE settings (optional)
LATENT_MOE_ENABLE="${LATENT_MOE_ENABLE:-True}"
LATENT_MOE_NUM_EXPERTS="${LATENT_MOE_NUM_EXPERTS:-4}"
LATENT_MOE_TOP_K="${LATENT_MOE_TOP_K:-2}"
LATENT_MOE_USE_SHARED_EXPERT="${LATENT_MOE_USE_SHARED_EXPERT:-True}"
LATENT_MOE_BALANCE_LOSS_WEIGHT="${LATENT_MOE_BALANCE_LOSS_WEIGHT:-0.1}"
LATENT_MOE_STEP_EMBED_MAX_STEPS="${LATENT_MOE_STEP_EMBED_MAX_STEPS:-32}"
LATENT_MOE_CONTEXT_TYPE="${LATENT_MOE_CONTEXT_TYPE:-disc}"
LATENT_MOE_EXPERT_DROPOUT="${LATENT_MOE_EXPERT_DROPOUT:-0.1}"
export LATENT_MOE_ENABLE LATENT_MOE_NUM_EXPERTS LATENT_MOE_TOP_K LATENT_MOE_USE_SHARED_EXPERT
export LATENT_MOE_BALANCE_LOSS_WEIGHT LATENT_MOE_STEP_EMBED_MAX_STEPS LATENT_MOE_CONTEXT_TYPE LATENT_MOE_EXPERT_DROPOUT

NNODES=${#NODES[@]}

echo "============================================="
echo "[LAUNCH] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[LAUNCH] Nodes (${NNODES}): ${NODES[*]}"
echo "[LAUNCH] MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "[LAUNCH] WANDB mode=${WANDB_MODE}, project=${WANDB_PROJECT}, name=${WANDB_NAME}, group=${WANDB_RUN_GROUP}, resume=${WANDB_RESUME}"
echo "[LAUNCH] latent_moe enable=${LATENT_MOE_ENABLE}, experts=${LATENT_MOE_NUM_EXPERTS}, top_k=${LATENT_MOE_TOP_K}, ctx=${LATENT_MOE_CONTEXT_TYPE}, balance_w=${LATENT_MOE_BALANCE_LOSS_WEIGHT}, expert_dropout=${LATENT_MOE_EXPERT_DROPOUT}"
echo "[LAUNCH] deepspeed use=${USE_DEEPSPEED}, cfg=${DEEPSPEED_CFG}"
echo "[LAUNCH] comm_profile=${COMM_PROFILE}"
echo "============================================="

# Ensure output dir exists
mkdir -p "${WORK_DIR}/${OUTPUT_DIR}"

# Launch worker nodes 1..N-1 via SSH (background, detached)
for (( rank=1; rank<NNODES; rank++ )); do
    ip="${NODES[$rank]}"
    log_file="${WORK_DIR}/${OUTPUT_DIR}/node${rank}.log"

    echo "[LAUNCH] Starting node ${rank} on ${ip} ..."

    ssh -f -o StrictHostKeyChecking=no "${PLUME_SSH_USER:-$USER}@${ip}" \
        "source ${CONDA_BASE}/etc/profile.d/conda.sh && \
         conda activate ${CONDA_ENV} && \
         cd ${WORK_DIR} && \
         export NODE_RANK=${rank} OUTPUT_DIR=${OUTPUT_DIR} NNODES=${NNODES} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RESUME_CKPT=${RESUME_CKPT} USE_DEEPSPEED=${USE_DEEPSPEED} DEEPSPEED_CFG=${DEEPSPEED_CFG} COMM_PROFILE=${COMM_PROFILE} WANDB_MODE=${WANDB_MODE} WANDB_PROJECT=${WANDB_PROJECT} WANDB_NAME=${WANDB_NAME} WANDB_RUN_GROUP=${WANDB_RUN_GROUP} WANDB_RESUME=${WANDB_RESUME} WANDB_TAGS=${WANDB_TAGS} LATENT_MOE_ENABLE=${LATENT_MOE_ENABLE} LATENT_MOE_NUM_EXPERTS=${LATENT_MOE_NUM_EXPERTS} LATENT_MOE_TOP_K=${LATENT_MOE_TOP_K} LATENT_MOE_USE_SHARED_EXPERT=${LATENT_MOE_USE_SHARED_EXPERT} LATENT_MOE_BALANCE_LOSS_WEIGHT=${LATENT_MOE_BALANCE_LOSS_WEIGHT} LATENT_MOE_STEP_EMBED_MAX_STEPS=${LATENT_MOE_STEP_EMBED_MAX_STEPS} LATENT_MOE_CONTEXT_TYPE=${LATENT_MOE_CONTEXT_TYPE} LATENT_MOE_EXPERT_DROPOUT=${LATENT_MOE_EXPERT_DROPOUT} && \
         nohup bash ${SCRIPT} > ${log_file} 2>&1 &"

    echo "[LAUNCH] Node ${rank} on ${ip} started (log: ${log_file})"
done

echo "[LAUNCH] Waiting 5s for workers to initialize..."
sleep 5

# Verify workers are running
for (( rank=1; rank<NNODES; rank++ )); do
    ip="${NODES[$rank]}"
    count=$(ssh -o StrictHostKeyChecking=no "${PLUME_SSH_USER:-$USER}@${ip}" \
        "ps aux | grep train_plume_gc | grep -v grep | wc -l" 2>/dev/null || echo "0")
    echo "[LAUNCH] Node ${rank} (${ip}): ${count} processes"
done

echo ""
echo "[LAUNCH] Now starting master (node 0) in foreground..."
echo "[LAUNCH] Press Ctrl+C to stop master (workers will also stop eventually)"
echo ""

# Master node 0 runs in foreground
export NODE_RANK=0
export NNODES=${NNODES}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export USE_DEEPSPEED=${USE_DEEPSPEED}
export DEEPSPEED_CFG=${DEEPSPEED_CFG}
export COMM_PROFILE=${COMM_PROFILE}
export WANDB_MODE=${WANDB_MODE}
export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_NAME=${WANDB_NAME}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP}
export WANDB_RESUME=${WANDB_RESUME}
export WANDB_TAGS=${WANDB_TAGS}
export LATENT_MOE_ENABLE=${LATENT_MOE_ENABLE}
export LATENT_MOE_NUM_EXPERTS=${LATENT_MOE_NUM_EXPERTS}
export LATENT_MOE_TOP_K=${LATENT_MOE_TOP_K}
export LATENT_MOE_USE_SHARED_EXPERT=${LATENT_MOE_USE_SHARED_EXPERT}
export LATENT_MOE_BALANCE_LOSS_WEIGHT=${LATENT_MOE_BALANCE_LOSS_WEIGHT}
export LATENT_MOE_STEP_EMBED_MAX_STEPS=${LATENT_MOE_STEP_EMBED_MAX_STEPS}
export LATENT_MOE_CONTEXT_TYPE=${LATENT_MOE_CONTEXT_TYPE}
export LATENT_MOE_EXPERT_DROPOUT=${LATENT_MOE_EXPERT_DROPOUT}
exec bash "${WORK_DIR}/${SCRIPT}"
