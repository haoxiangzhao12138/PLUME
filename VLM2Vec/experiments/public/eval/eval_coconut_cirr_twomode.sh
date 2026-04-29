#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export prefix=/home/guohaiyun/yangtianyu
#   export MODEL_NAME=$prefix/UME-R1/output/UME-R1-2B-Coconut-smoke-1step-2026-02-14-15-48-27/checkpoint-300
#   bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_cirr_twomode.sh

prefix="${prefix:-your_path}"
UME_ROOT="${UME_ROOT:-$prefix/UME-R1}"

MODEL_NAME="${MODEL_NAME:-$UME_ROOT/output/UME-R1-2B-Coconut-smoke-1step-2026-02-14-15-48-27/checkpoint-300}"
MODEL_BACKBONE="${MODEL_BACKBONE:-qwen2_vl}"
MODEL_BASE="${MODEL_BASE:-}"

MODE="${MODE:-gen}"  # gen or disc
DATASET_NAMES="${DATASET_NAMES:-CIRR}"  # comma-separated dataset keys in yaml
DATA_CONFIG_PATH="${DATA_CONFIG_PATH:-$UME_ROOT/src/eval/VLM2Vec/experiments/public/eval/image.yaml}"
DATA_BASEDIR="${DATA_BASEDIR:-/home/share/yty_data/vlm2vec_eval/MMEB-V2}"
USE_COCONUT_LATENT_REASONING="${USE_COCONUT_LATENT_REASONING:-True}"
COCONUT_LATENT_STEPS="${COCONUT_LATENT_STEPS:-auto}"
COCONUT_PREFIX_TEXT="${COCONUT_PREFIX_TEXT:-<think><bot>}"
COCONUT_FORCED_SUFFIX_TEXT="${COCONUT_FORCED_SUFFIX_TEXT:-<eot></think><answer>}"
FORCE_REEVAL="${FORCE_REEVAL:-False}"  # True: remove cached outputs for selected datasets

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
# Align eval preprocessing with training defaults in train/argument.py:
# max_pixels=28*28*576=451584, min_pixels=28*28*16=12544
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-451584}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-12544}"
# Align with training --model_max_length 4096
MAX_LEN="${MAX_LEN:-4096}"

RUN_TAG="${RUN_TAG:-$(basename "$MODEL_NAME")}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-image-cirr-$MODE-coconut}"
OUTPUT_PATH="${OUTPUT_PATH:-$UME_ROOT/output/$RUN_TAG/$OUTPUT_SUFFIX}"
LOG_DIR="${LOG_DIR:-$prefix/eval_log}"
mkdir -p "$OUTPUT_PATH" "$LOG_DIR"
LOG_PATH="$LOG_DIR/${RUN_TAG}_${MODE}_cirr.log"

cd "$UME_ROOT/src/eval/VLM2Vec" || exit 1

extra_args=()
if [[ -n "$MODEL_BASE" ]]; then
  extra_args+=(--model_base "$MODEL_BASE")
fi

if [[ "${USE_COCONUT_LATENT_REASONING,,}" == "true" ]] && [[ "$COCONUT_LATENT_STEPS" == "auto" ]]; then
  trainer_state_path="$MODEL_NAME/trainer_state.json"
  if [[ -f "$trainer_state_path" ]]; then
    COCONUT_LATENT_STEPS="$(
      python - "$trainer_state_path" <<'PY'
import json
import sys

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
PY
    )"
    echo "[AUTO] Detected COCONUT_LATENT_STEPS=$COCONUT_LATENT_STEPS from $trainer_state_path"
  else
    COCONUT_LATENT_STEPS=0
    echo "[WARN] trainer_state.json not found at $trainer_state_path; fallback COCONUT_LATENT_STEPS=$COCONUT_LATENT_STEPS"
  fi
fi

if [[ "${FORCE_REEVAL,,}" == "true" ]]; then
  IFS=',' read -r -a __dataset_arr <<< "$DATASET_NAMES"
  for __raw_name in "${__dataset_arr[@]}"; do
    __name="$(echo "$__raw_name" | xargs)"
    [[ -z "$__name" ]] && continue
    for __path in \
      "$OUTPUT_PATH/${__name}_${MODE}_${MODE}_score.json" \
      "$OUTPUT_PATH/${__name}_${MODE}_${MODE}_pred.jsonl" \
      "$OUTPUT_PATH/${__name}_${MODE}_${MODE}_info.jsonl" \
      "$OUTPUT_PATH/${__name}_${MODE}_qry" \
      "$OUTPUT_PATH/${__name}_${MODE}_tgt"; do
      if [[ -e "$__path" ]]; then
        rm -rf "$__path"
        echo "[FORCE_REEVAL] Removed cache: $__path"
      fi
    done
  done
fi

echo "================================================="
echo "Model      : $MODEL_NAME"
echo "Backbone   : $MODEL_BACKBONE"
echo "Datasets   : $DATASET_NAMES"
echo "Mode       : $MODE"
echo "Latent     : $USE_COCONUT_LATENT_REASONING (steps=$COCONUT_LATENT_STEPS)"
echo "Output     : $OUTPUT_PATH"
echo "Log        : $LOG_PATH"
echo "================================================="

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  eval_twomode.py \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --model_backbone "$MODEL_BACKBONE" \
  --model_name "$MODEL_NAME" \
  --dataset_config "$DATA_CONFIG_PATH" \
  --dataset_names "$DATASET_NAMES" \
  --encode_output_path "$OUTPUT_PATH" \
  --data_basedir "$DATA_BASEDIR" \
  --max_len "$MAX_LEN" \
  --resize_max_pixels "$RESIZE_MAX_PIXELS" \
  --resize_min_pixels "$RESIZE_MIN_PIXELS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --qry_mode "$MODE" \
  --tgt_mode "$MODE" \
  --use_coconut_latent_reasoning "$USE_COCONUT_LATENT_REASONING" \
  --coconut_latent_steps "$COCONUT_LATENT_STEPS" \
  --coconut_prefix_text "$COCONUT_PREFIX_TEXT" \
  --coconut_forced_suffix_text "$COCONUT_FORCED_SUFFIX_TEXT" \
  "${extra_args[@]}" \
  2>&1 | tee -a "$LOG_PATH"

echo "Done. Log saved to: $LOG_PATH"
