#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Overridable locations (defaults match this machine's current layout).
RAGMARK_DATA_DIR="${RAGMARK_DATA_DIR:-/mnt/nvme1n1p1/zman/ragmark_data}"
DOCKER_IMAGE="${DOCKER_IMAGE:-ragmark:local}"
TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${RAGMARK_DATA_DIR}/tiktoken_cache}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[AE Eval] HF_TOKEN is not set in this shell. Llama checkpoints are gated and will fail to download." >&2
  echo "[AE Eval] export HF_TOKEN=<your token> (open a new terminal first if you've only added it to ~/.bashrc)." >&2
  exit 1
fi

if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "[AE Eval] Docker image '$DOCKER_IMAGE' not found locally. Build it first:" >&2
  echo "[AE Eval]   docker build -t $DOCKER_IMAGE $ROOT_DIR" >&2
  exit 1
fi

if [[ ! -d "$TIKTOKEN_CACHE_DIR" ]]; then
  echo "[AE Eval] Warning: $TIKTOKEN_CACHE_DIR not found." >&2
  echo "[AE Eval] The compression sweep (Fig. 5) needs a pre-seeded tiktoken cache if" >&2
  echo "[AE Eval] openaipublic.blob.core.windows.net is blocked on this network." >&2
fi

echo "[AE Eval] Root: $ROOT_DIR"
echo "[AE Eval] Data dir: $RAGMARK_DATA_DIR"
echo "[AE Eval] Docker image: $DOCKER_IMAGE"

run_in_docker() {
  local script="$1"
  echo
  echo "[AE Eval] ==================================================================="
  echo "[AE Eval] Running $script"
  echo "[AE Eval] ==================================================================="
  docker run --rm --gpus all \
    -v "${RAGMARK_DATA_DIR}:${RAGMARK_DATA_DIR}" \
    -v "${ROOT_DIR}:/app" \
    -v "${ROOT_DIR}:${ROOT_DIR}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR}" \
    "$DOCKER_IMAGE" bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate ragmark && python -u ${script}"
}

run_in_docker "ae_evaluation/run_sweep_fig4_fig9.py"
run_in_docker "ae_evaluation/run_sweep_fig5.py"

JUPYTER_BIN="${ROOT_DIR}/.venv/bin/jupyter"
if [[ ! -x "$JUPYTER_BIN" ]]; then
  echo "[AE Eval] $JUPYTER_BIN not found — install jupyter/pandas/matplotlib into .venv first (see README First-run setup)." >&2
  exit 1
fi

echo
echo "[AE Eval] ==================================================================="
echo "[AE Eval] Running load_evals_executed.ipynb"
echo "[AE Eval] ==================================================================="
"$JUPYTER_BIN" nbconvert --to notebook --execute --inplace "${ROOT_DIR}/ae_evaluation/load_evals_executed.ipynb"

echo
echo "[AE Eval] Done. Total sweep times written to:"
echo "  ${ROOT_DIR}/bench_raw_data/fig4_fig9_total_sweep_time.txt"
echo "  ${ROOT_DIR}/bench_raw_data/fig5_total_sweep_time.txt"
echo "[AE Eval] Figures and data written to ${ROOT_DIR}/ae_evaluation/figures/"
