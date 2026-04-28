#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# COCONUT Latent Reasoning — Full Multi-Modal Evaluation
# Evaluates image / video / visdoc modalities with latent reasoning support.
#
# Usage:
#   bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh
#
# Override any variable via environment, e.g.:
#   CHECKPOINT=output/xxx/checkpoint-400 bash eval_coconut_all_modalities.sh
# ==============================================================================
#MODALITIES=image \
# MODALITIES=image DATASET_NAMES="InfographicsVQA" \
# DATASET_NAMES="ImageNet-1K,N24News,HatefulMemes,VOC2007,SUN397,Place365,ImageNet-A,ImageNet-R,ObjectNet,Country211,OK-VQA,A-OKVQA,DocVQA,InfographicsVQA" \
# DATASET_NAMES="ChartQA,Visual7W,ScienceQA,VizWiz,GQA,TextVQA,VisDial,CIRR,MSCOCO_t2i,MSCOCO_i2t"
# DATASET_NAMES="Wiki-SS-NQ,VisualNews_t2i,VisualNews_i2t,FashionIQ,Visual7W-Pointing" \
# DATASET_NAMES="NIGHTS,WebQA,OVEN,EDIS,MSCOCO,RefCOCO,RefCOCO-Matching" \
# bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh\


# 无ans：
# CHECKPOINT=/home/guohaiyun/yangtianyu/UME-R1/output/UME-R1-2B-Coconut-Fulldata-NoAns-4node-2026-03-10-10-03-52/checkpoint-1431 \
# USE_COCONUT_LATENT_REASONING=True \
# COCONUT_LATENT_STEPS=4 \
# COCONUT_FORCED_SUFFIX_TEXT="<eot></think><gen_emb>" \
# bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh
# ---------- Paths ----------
prefix="${prefix:-/home/guohaiyun/yangtianyu}"
UME_ROOT="${UME_ROOT:-$prefix/UME-R1}"

CHECKPOINT="${CHECKPOINT:-$UME_ROOT/output/UME-R1-2B-Coconut-Fulldata-NoAns-4node-2026-03-10-10-03-52/checkpoint-1431}"
# MODEL_BASE: set to empty string to skip (for standalone models), or provide a base model path
if [[ -z "${MODEL_BASE+x}" ]]; then
    # MODEL_BASE not set, use default
    MODEL_BASE="/home/share/yty_model/Qwen/Qwen2-VL-2B-Instruct"
fi
# If MODEL_BASE is explicitly set to empty, keep it empty
MODEL_BACKBONE="${MODEL_BACKBONE:-qwen2_vl}"

DATA_BASEDIR="${DATA_BASEDIR:-/home/share/yty_data/vlm2vec_eval/MMEB-V2}"
OUTPUT_BASEDIR="${OUTPUT_BASEDIR:-$UME_ROOT/output/Eval/UME-R1_2B}"
LOG_DIR="${LOG_DIR:-$prefix/eval_log}"
                                                                                                                            
# ---------- COCONUT Latent Config ----------
USE_COCONUT_LATENT_REASONING="${USE_COCONUT_LATENT_REASONING:-True}"
# "auto" = read from trainer_state.json; or set a fixed integer
COCONUT_LATENT_STEPS="${COCONUT_LATENT_STEPS:-4}"
COCONUT_PREFIX_TEXT="${COCONUT_PREFIX_TEXT:-<think><bot>}"
COCONUT_FORCED_SUFFIX_TEXT="${COCONUT_FORCED_SUFFIX_TEXT:=<eot></think>
<gen_emb>}"
# Enable detailed token logging (may slow down evaluation)
DEBUG_LOG_TOKENS="${DEBUG_LOG_TOKENS:-False}"

# ---------- Eval Config ----------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Latent reasoning requires batch_size=1 for gen mode
GEN_BATCH_SIZE=1
# disc mode can use larger batch
DISC_BATCH_SIZE="${DISC_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
# Align with training defaults (28*28*576=451584, 28*28*16=12544)
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2359296}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-784}"
MAX_LEN="${MAX_LEN:-11288}"

FORCE_REEVAL="${FORCE_REEVAL:-False}"

# Filter specific datasets (comma-separated). Empty = all datasets in yaml.
DATASET_NAMES="${DATASET_NAMES:-}"

# Modalities and their yaml configs
# Override via env: MODALITIES="image" or MODALITIES="image,video"
if [[ -n "${MODALITIES:-}" ]]; then
    IFS=',' read -r -a MODALITIES <<< "${MODALITIES}"
else
    declare -a MODALITIES=("image" "video" "visdoc")
fi

# qry_mode / tgt_mode: gen or disc
QRY_MODE="${QRY_MODE:-gen}"
TGT_MODE="${TGT_MODE:-gen}"

# ==============================================================================
# Auto-detect latent steps from trainer_state.json
# ==============================================================================
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

# Determine if latent reasoning should actually be enabled
# (if latent_steps=0 and use_coconut=True, the latent loop is a no-op but
#  the prefix/suffix injection and <gen_emb> extraction still apply)
USE_LATENT_FLAG="${USE_COCONUT_LATENT_REASONING}"

# ==============================================================================
# Derived paths
# ==============================================================================
RUN_TAG="$(basename "$(dirname "${CHECKPOINT}")")/$(basename "${CHECKPOINT}")"
RUN_TAG_FLAT="$(echo "${RUN_TAG}" | tr '/' '_')"

cd "${UME_ROOT}/src/eval/VLM2Vec" || exit 1
mkdir -p "${LOG_DIR}"

# ==============================================================================
# Main loop
# ==============================================================================
echo "============================================================"
echo "  COCONUT Multi-Modal Evaluation"
echo "============================================================"
echo "  Checkpoint    : ${CHECKPOINT}"
echo "  Base model    : ${MODEL_BASE}"
echo "  Backbone      : ${MODEL_BACKBONE}"
if [[ "${USE_LATENT_FLAG,,}" == "true" ]]; then
    echo "  Latent        : ${USE_LATENT_FLAG} (steps=${LATENT_STEPS})"
    echo "  Prefix        : ${COCONUT_PREFIX_TEXT}"
    echo "  Suffix        : ${COCONUT_FORCED_SUFFIX_TEXT}"
    echo "  Debug logging : ${DEBUG_LOG_TOKENS}"
else
    echo "  Latent        : ${USE_LATENT_FLAG} (disabled, latent params ignored)"
fi
echo "  Modalities    : ${MODALITIES[*]}"
echo "  qry/tgt mode  : ${QRY_MODE}/${TGT_MODE}"
echo "  GPUs          : ${CUDA_VISIBLE_DEVICES} (${NPROC_PER_NODE} procs)"
echo "  Resize pixels : ${RESIZE_MIN_PIXELS} ~ ${RESIZE_MAX_PIXELS}"
echo "============================================================"

for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="${UME_ROOT}/src/eval/VLM2Vec/experiments/public/eval/${MODALITY}.yaml"
    OUTPUT_PATH="${OUTPUT_BASEDIR}/${RUN_TAG_FLAT}/${MODALITY}-${QRY_MODE}"
    LOG_PATH="${LOG_DIR}/${RUN_TAG_FLAT}_${MODALITY}_${QRY_MODE}.log"

    if [[ ! -f "${DATA_CONFIG_PATH}" ]]; then
        echo "[SKIP] Config not found: ${DATA_CONFIG_PATH}"
        continue
    fi

    mkdir -p "${OUTPUT_PATH}"

    # Determine batch size: latent gen mode requires bs=1
    if [[ "${QRY_MODE}" == "gen" ]] && [[ "${USE_LATENT_FLAG,,}" == "true" ]]; then
        BATCH_SIZE="${GEN_BATCH_SIZE}"
    else
        BATCH_SIZE="${DISC_BATCH_SIZE}"
    fi

    # Force re-eval: remove cached outputs
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
        "${extra_args[@]}" \
        2>&1 | tee -a "${LOG_PATH}"

    echo "  ✅ ${MODALITY} done."
done

echo ""
echo "============================================================"
echo "  ✅ All modalities evaluated."
echo "  Results in: ${OUTPUT_BASEDIR}/${RUN_TAG_FLAT}/"
echo "============================================================"
