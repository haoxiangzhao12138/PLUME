#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# COCONUT Latent-MoE — Full Multi-Modal Evaluation
# Evaluates image / video / visdoc modalities with latent reasoning + MoE support.
#
# Usage:
#   bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_moe_all_modalities.sh
#
# Override any variable via environment, e.g.:
#   CHECKPOINT=output/xxx/checkpoint-400 bash eval_coconut_moe_all_modalities.sh
#
# Ablate one or more routed experts (router never selects them; top-k is taken from the rest):
#   COCONUT_LATENT_MOE_EXCLUDE_EXPERTS=2 bash ...   # drop expert index 2 only
#   COCONUT_LATENT_MOE_EXCLUDE_EXPERTS=0,3 bash ... # drop experts 0 and 3
# Output dir gets a suffix like _exc2 or _exc0-3 so runs do not overwrite.


# CHECKPOINT=/path/to/your/moe/checkpoint \
# USE_COCONUT_LATENT_REASONING=True \
# USE_COCONUT_LATENT_MOE=True \
# COCONUT_LATENT_MOE_NUM_EXPERTS=4 \
# COCONUT_LATENT_MOE_TOP_K=2 \
# COCONUT_LATENT_MOE_USE_SHARED_EXPERT=True \
# COCONUT_LATENT_MOE_STEP_EMBED_MAX_STEPS=32 \
# COCONUT_LATENT_MOE_CONTEXT_TYPE=prefix_last \
# bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_moe_all_modalities.sh
# ==============================================================================

# ---------- Paths ----------
prefix="${prefix:-/home/guohaiyun/yangtianyu}"
UME_ROOT="${UME_ROOT:-$prefix/UME-R1}"

CHECKPOINT="${CHECKPOINT:-$UME_ROOT/output/eval-test/UME-R1-2B-EvalImage-NoAns-2026-03-28-16-59-17/checkpoint-800}"
if [[ -z "${MODEL_BASE+x}" ]]; then
    MODEL_BASE="/home/share/yty_model/UME-R1/2B/UME-R1/2B"
fi
MODEL_BACKBONE="${MODEL_BACKBONE:-qwen2_vl}"

DATA_BASEDIR="${DATA_BASEDIR:-/home/share/yty_data/vlm2vec_eval/MMEB-V2}"
OUTPUT_BASEDIR="${OUTPUT_BASEDIR:-$UME_ROOT/output/Eval/UME-R1_2B_test-8latent}"
LOG_DIR="${LOG_DIR:-$prefix/eval_log}"

# ---------- COCONUT Latent Config ----------
USE_COCONUT_LATENT_REASONING="${USE_COCONUT_LATENT_REASONING:-True}"
COCONUT_LATENT_STEPS="${COCONUT_LATENT_STEPS:-8}"
COCONUT_PREFIX_TEXT="${COCONUT_PREFIX_TEXT:-<think><bot>}"
# COCONUT_FORCED_SUFFIX_TEXT="${COCONUT_FORCED_SUFFIX_TEXT:-$'<eot></think>\n<gen_emb>'}"
COCONUT_FORCED_SUFFIX_TEXT="${COCONUT_FORCED_SUFFIX_TEXT:-$'<eot></think>\n<gen_emb>'}"

DEBUG_LOG_TOKENS="${DEBUG_LOG_TOKENS:-False}"

# ---------- COCONUT Latent MoE Config ----------
USE_COCONUT_LATENT_MOE="${USE_COCONUT_LATENT_MOE:-True}"
COCONUT_LATENT_MOE_NUM_EXPERTS="${COCONUT_LATENT_MOE_NUM_EXPERTS:-4}"
COCONUT_LATENT_MOE_TOP_K="${COCONUT_LATENT_MOE_TOP_K:-2}"
COCONUT_LATENT_MOE_USE_SHARED_EXPERT="${COCONUT_LATENT_MOE_USE_SHARED_EXPERT:-True}"
COCONUT_LATENT_MOE_STEP_EMBED_MAX_STEPS="${COCONUT_LATENT_MOE_STEP_EMBED_MAX_STEPS:-32}"
COCONUT_LATENT_MOE_CONTEXT_TYPE="${COCONUT_LATENT_MOE_CONTEXT_TYPE:-disc}"
# Comma-separated routed expert indices (0..num_experts-1) to mask out at eval; empty = no mask.
COCONUT_LATENT_MOE_EXCLUDE_EXPERTS="${COCONUT_LATENT_MOE_EXCLUDE_EXPERTS:-}"

# ---------- Eval Config ---------- 
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GEN_BATCH_SIZE=1
DISC_BATCH_SIZE="${DISC_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2359296}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-784}"
MAX_LEN="${MAX_LEN:-11288}"

FORCE_REEVAL="${FORCE_REEVAL:-False}"
DATASET_NAMES="${DATASET_NAMES:-}"

if [[ -n "${MODALITIES:-}" ]]; then
    IFS=',' read -r -a MODALITIES <<< "${MODALITIES}"
else
    declare -a MODALITIES=("image" "video" "visdoc")
fi

QRY_MODE="${QRY_MODE:-gen}"
TGT_MODE="${TGT_MODE:-gen}"

resolve_latent_steps() {
    local ckpt_path="$1"
    local requested="$2"

    if [[ "${requested}" != "auto" ]]; then
        echo "${requested}"
        return
    fi

    local trainer_state_path="${ckpt_path}/trainer_state.json"
    if [[ ! -f "${trainer_state_path}" ]]; then
        echo "[WARN] trainer_state.json not found at ${trainer_state_path}; fallback to 0" >&2
        echo "0"
        return
    fi

    local steps
    steps="$(python3 - "${trainer_state_path}" <<'PYEOF'
import json, sys
path = sys.argv[1]
latent = 0
with open(path, "r", encoding="utf-8") as f:
    state = json.load(f)
for item in reversed(state.get("log_history", [])):
    if "curriculum_latent_tokens" in item:
        try:
            latent = int(round(float(item["curriculum_latent_tokens"])))
        except Exception:
            latent = 0
        break
print(latent)
PYEOF
    )"
    echo "[AUTO] Detected COCONUT_LATENT_STEPS=${steps} from ${trainer_state_path}" >&2
    echo "${steps}"
}

LATENT_STEPS="$(resolve_latent_steps "${CHECKPOINT}" "${COCONUT_LATENT_STEPS}")"
USE_LATENT_FLAG="${USE_COCONUT_LATENT_REASONING}"
USE_LATENT_MOE_FLAG="${USE_COCONUT_LATENT_MOE}"

RUN_TAG="$(basename "$(dirname "${CHECKPOINT}")")/$(basename "${CHECKPOINT}")"
RUN_TAG_FLAT="$(echo "${RUN_TAG}" | tr '/' '_')"
if [[ -n "${COCONUT_LATENT_MOE_EXCLUDE_EXPERTS}" ]]; then
    RUN_TAG_FLAT="${RUN_TAG_FLAT}_exc$(echo "${COCONUT_LATENT_MOE_EXCLUDE_EXPERTS}" | tr ',' '-')"
fi

cd "${UME_ROOT}/src/eval/VLM2Vec" || exit 1
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  COCONUT Latent-MoE Multi-Modal Evaluation"
echo "============================================================"
echo "  Checkpoint    : ${CHECKPOINT}"
echo "  Base model    : ${MODEL_BASE}"
echo "  Backbone      : ${MODEL_BACKBONE}"
if [[ "${USE_LATENT_FLAG,,}" == "true" ]]; then
    echo "  Latent        : ${USE_LATENT_FLAG} (steps=${LATENT_STEPS})"
    echo "  Prefix        : ${COCONUT_PREFIX_TEXT}"
    echo "  Suffix        : ${COCONUT_FORCED_SUFFIX_TEXT}"
    echo "  Debug logging : ${DEBUG_LOG_TOKENS}"
    if [[ "${USE_LATENT_MOE_FLAG,,}" == "true" ]]; then
        echo "  Latent-MoE    : ${USE_LATENT_MOE_FLAG}"
        echo "    num_experts : ${COCONUT_LATENT_MOE_NUM_EXPERTS}"
        echo "    top_k       : ${COCONUT_LATENT_MOE_TOP_K}"
        echo "    shared      : ${COCONUT_LATENT_MOE_USE_SHARED_EXPERT}"
        echo "    step_max    : ${COCONUT_LATENT_MOE_STEP_EMBED_MAX_STEPS}"
        echo "    context     : ${COCONUT_LATENT_MOE_CONTEXT_TYPE}"
        if [[ -n "${COCONUT_LATENT_MOE_EXCLUDE_EXPERTS}" ]]; then
            echo "    exclude_experts: ${COCONUT_LATENT_MOE_EXCLUDE_EXPERTS}"
        else
            echo "    exclude_experts: (none)"
        fi
    else
        echo "  Latent-MoE    : ${USE_LATENT_MOE_FLAG} (disabled)"
    fi
else
    echo "  Latent        : ${USE_LATENT_FLAG} (disabled, latent/moe params ignored)"
fi
echo "  Modalities    : ${MODALITIES[*]}"
echo "  qry/tgt mode  : ${QRY_MODE}/${TGT_MODE}"
echo "  GPUs          : ${CUDA_VISIBLE_DEVICES} (${NPROC_PER_NODE} procs)"
echo "  Resize pixels : ${RESIZE_MIN_PIXELS} ~ ${RESIZE_MAX_PIXELS}"
echo "============================================================"

for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="${UME_ROOT}/src/eval/VLM2Vec/experiments/public/eval/${MODALITY}.yaml"
    OUTPUT_PATH="${OUTPUT_BASEDIR}/${RUN_TAG_FLAT}/${MODALITY}-${QRY_MODE}"
    LOG_PATH="${LOG_DIR}/${RUN_TAG_FLAT}_${MODALITY}_${QRY_MODE}_moe.log"

    if [[ ! -f "${DATA_CONFIG_PATH}" ]]; then
        echo "[SKIP] Config not found: ${DATA_CONFIG_PATH}"
        continue
    fi

    mkdir -p "${OUTPUT_PATH}"

    if [[ "${QRY_MODE}" == "gen" ]] && [[ "${USE_LATENT_FLAG,,}" == "true" ]]; then
        BATCH_SIZE="${GEN_BATCH_SIZE}"
    else
        BATCH_SIZE="${DISC_BATCH_SIZE}"
    fi

    if [[ "${FORCE_REEVAL,,}" == "true" ]]; then
        echo "[FORCE_REEVAL] Cleaning cached outputs in ${OUTPUT_PATH}..."
        find "${OUTPUT_PATH}" -maxdepth 1 \( -name "*_score.json" -o -name "*_pred.jsonl" -o -name "*_info.jsonl" -o -name "*_qry" -o -name "*_tgt" \) -exec rm -rf {} +
    fi

    echo ""
    echo "---------------------------------------------------------"
    echo "  Modality   : ${MODALITY}"
    echo "  Config     : ${DATA_CONFIG_PATH}"
    echo "  Output     : ${OUTPUT_PATH}"
    echo "  Batch size : ${BATCH_SIZE}"
    echo "  Log        : ${LOG_PATH}"
    echo "---------------------------------------------------------"

    extra_args=()
    if [[ -n "${MODEL_BASE}" ]]; then
        extra_args+=(--model_base "${MODEL_BASE}")
    fi
    if [[ -n "${DATASET_NAMES}" ]]; then
        extra_args+=(--dataset_names "${DATASET_NAMES}")
    fi

    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    torchrun \
        --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" \
        eval_twomode.py \
        --per_device_eval_batch_size "${BATCH_SIZE}" \
        --model_backbone "${MODEL_BACKBONE}" \
        --model_name "${CHECKPOINT}" \
        --dataset_config "${DATA_CONFIG_PATH}" \
        --encode_output_path "${OUTPUT_PATH}" \
        --data_basedir "${DATA_BASEDIR}" \
        --max_len "${MAX_LEN}" \
        --resize_max_pixels "${RESIZE_MAX_PIXELS}" \
        --resize_min_pixels "${RESIZE_MIN_PIXELS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --qry_mode "${QRY_MODE}" \
        --tgt_mode "${TGT_MODE}" \
        --use_coconut_latent_reasoning "${USE_LATENT_FLAG}" \
        --coconut_latent_steps "${LATENT_STEPS}" \
        --coconut_prefix_text "${COCONUT_PREFIX_TEXT}" \
        --coconut_forced_suffix_text "${COCONUT_FORCED_SUFFIX_TEXT}" \
        --debug_log_tokens "${DEBUG_LOG_TOKENS}" \
        --use_coconut_latent_moe "${USE_LATENT_MOE_FLAG}" \
        --coconut_latent_moe_num_experts "${COCONUT_LATENT_MOE_NUM_EXPERTS}" \
        --coconut_latent_moe_top_k "${COCONUT_LATENT_MOE_TOP_K}" \
        --coconut_latent_moe_use_shared_expert "${COCONUT_LATENT_MOE_USE_SHARED_EXPERT}" \
        --coconut_latent_moe_step_embed_max_steps "${COCONUT_LATENT_MOE_STEP_EMBED_MAX_STEPS}" \
        --coconut_latent_moe_context_type "${COCONUT_LATENT_MOE_CONTEXT_TYPE}" \
        --coconut_latent_moe_exclude_experts "${COCONUT_LATENT_MOE_EXCLUDE_EXPERTS}" \
        "${extra_args[@]}" \
        2>&1 | tee -a "${LOG_PATH}"

    echo "  ✅ ${MODALITY} done."
done

echo ""
echo "============================================================"
echo "  ✅ All modalities evaluated."
echo "  Results in: ${OUTPUT_BASEDIR}/${RUN_TAG_FLAT}/"
echo "============================================================"
