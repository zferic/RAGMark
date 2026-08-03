#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[AE Quick] Neither python nor python3 found in PATH" >&2
  exit 1
fi

# Read machine-local defaults from config.py to avoid hardcoding infra details.
eval "$("$PYTHON_BIN" - <<'PY'
from config import BASE_OUT, OUT_SUFFIX, WIKI_INDEX_DIR, WIKI_CORPUS_2018, DATASET_DIR
print(f'BASE_OUT={BASE_OUT!r}')
print(f'OUT_SUFFIX={OUT_SUFFIX!r}')
print(f'WIKI_INDEX_DIR={WIKI_INDEX_DIR!r}')
print(f'WIKI_CORPUS_2018={WIKI_CORPUS_2018!r}')
print(f'DATASET_DIR={DATASET_DIR!r}')
PY
)"

# Reviewer-overridable parameters.
MODEL="${AE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
RET_MODEL="${AE_RET_MODEL:-intfloat/e5-base-v2}"
DATASET_FILE="${AE_DATASET_FILE:-nq_dataset.jsonl}"
EVAL_SIZE="${AE_EVAL_SIZE:-20}"
BATCH_SIZE="${AE_BATCH_SIZE:-1}"
TOP_K="${AE_TOP_K:-5}"
PIPELINE="${AE_PIPELINE:-standard}"

EMBED_DEVICE="${AE_EMBED_DEVICE:-cuda:0}"
GEN_DEVICE="${AE_GEN_DEVICE:-cuda:0}"
INDEX_DEVICE="${AE_INDEX_DEVICE:-cpu}"
COMPRESS_DEVICE="${AE_COMPRESS_DEVICE:-cuda:0}"
RERANK_DEVICE="${AE_RERANK_DEVICE:-cuda:0}"

INDEX_PATH="${AE_INDEX_PATH:-${WIKI_INDEX_DIR}/e5-base-v2-2018/index/e5_Flat.index}"
CORPUS_PATH="${AE_CORPUS_PATH:-${WIKI_CORPUS_2018}}"
DATASET_PATH="${AE_DATASET_PATH:-${DATASET_DIR}/${DATASET_FILE}}"
OUT_SUFFIX_AE="${AE_OUT_SUFFIX:-${OUT_SUFFIX}_aequick}"

SWEEP_CONFIGS="[{\"use_rag\": true, \"compress\": false, \"compress_method\": \"none\", \"compression_rate\": 1.0, \"rag_top_k\": ${TOP_K}}]"
EVAL_PATHS="[\"${DATASET_PATH}\"]"

echo "[AE Quick] Root: $ROOT_DIR"
echo "[AE Quick] Model: $MODEL"
echo "[AE Quick] Retriever: $RET_MODEL"
echo "[AE Quick] Dataset: $DATASET_PATH"
echo "[AE Quick] Eval size: $EVAL_SIZE"
echo "[AE Quick] Devices: embed=$EMBED_DEVICE gen=$GEN_DEVICE index=$INDEX_DEVICE"
echo "[AE Quick] Output suffix: $OUT_SUFFIX_AE"

if [[ ! -f "$INDEX_PATH" ]]; then
  echo "[AE Quick] Missing FAISS index: $INDEX_PATH" >&2
  exit 1
fi
if [[ ! -f "$CORPUS_PATH" ]]; then
  echo "[AE Quick] Missing corpus JSONL: $CORPUS_PATH" >&2
  exit 1
fi
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[AE Quick] Missing dataset JSONL: $DATASET_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" -u run_sweep.py \
  --model_path "$MODEL" \
  --retrieval_model "$RET_MODEL" \
  --retrieval_index "$INDEX_PATH" \
  --retrieval_json "$CORPUS_PATH" \
  --eval_paths "$EVAL_PATHS" \
  --eval_size "$EVAL_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --base_out "$BASE_OUT" \
  --out_suffix "$OUT_SUFFIX_AE" \
  --sweep_configs "$SWEEP_CONFIGS" \
  --pipeline "$PIPELINE" \
  --embed_device "$EMBED_DEVICE" \
  --gen_device "$GEN_DEVICE" \
  --index_device "$INDEX_DEVICE" \
  --compress_device "$COMPRESS_DEVICE" \
  --rerank_device "$RERANK_DEVICE"

echo "[AE Quick] Done. Inspect outputs under:"
echo "  $BASE_OUT/testingevals$OUT_SUFFIX_AE"
echo "  $BASE_OUT/timingevals$OUT_SUFFIX_AE"
