# RAGMark Artifact Evaluation Guide (IISWC 2026)

This guide is for artifact reviewers using SSH access to a prepared evaluation machine.

RAGMark reproduces the paper's five characterization cases: naive, reranking, compression, combined, and iterative.

Scope:
- No cloud setup.
- No container registry workflows (the Docker image is built and run locally, never pulled from a registry).
- No large data download pipeline requirements.
- Use machine-local datasets, FAISS index files, and model cache already provisioned for evaluation.

## 1. Reviewer Quick Start

From the repository root:

```bash
bash ae_evaluation/ae_eval.sh
```

This runs the full artifact pipeline end-to-end:
1. Preflight checks: `HF_TOKEN` is set, the `ragmark:local` Docker image exists, and a tiktoken cache is present (see Section 7).
2. `ae_evaluation/run_sweep_fig4_fig9.py` inside the `ragmark:local` container (reranking + naive configs, feeding Fig. 4 and Fig. 9).
3. `ae_evaluation/run_sweep_fig5.py` inside the same container (compression-rate sweep, feeding Fig. 5).
4. `ae_evaluation/load_evals_executed.ipynb` executed headlessly via `.venv/bin/jupyter nbconvert`, which loads every output file and renders the Fig. 4 / 5 / 9 reproductions.

Each sweep's total wall-clock time is written to `bench_raw_data/{fig4_fig9,fig5}_total_sweep_time.txt`; figures and their underlying data tables land in `ae_evaluation/figures/{plots,data}/`.

Overridable environment variables (defaults match this machine's layout):
- `HF_TOKEN` — required; a Hugging Face token with accepted Meta license for the gated Llama checkpoints. No default.
- `RAGMARK_DATA_DIR` — host directory holding the FAISS index, corpus, datasets, HF cache, and tiktoken cache. Default `/mnt/nvme1n1p1/zman/ragmark_data`.
- `DOCKER_IMAGE` — image to run. Default `ragmark:local`.
- `TIKTOKEN_CACHE_DIR` — pre-seeded tiktoken cache directory (see Section 7). Default `${RAGMARK_DATA_DIR}/tiktoken_cache`.

To run either sweep on its own instead (e.g. while iterating), invoke it directly through the same Docker setup:

```bash
docker run --rm --gpus all \
  -v /mnt/nvme1n1p1/zman/ragmark_data:/mnt/nvme1n1p1/zman/ragmark_data \
  -v "$(pwd)":/app -v "$(pwd)":"$(pwd)" \
  -e HF_TOKEN="$HF_TOKEN" -e TIKTOKEN_CACHE_DIR=/mnt/nvme1n1p1/zman/ragmark_data/tiktoken_cache \
  ragmark:local bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate ragmark && python ae_evaluation/run_sweep_fig4_fig9.py"
```

**Note on the checked-in sweep grids:** `run_sweep_fig4_fig9.py` and `run_sweep_fig5.py` currently have `gen_models` scoped down to `Llama-3.2-1B-Instruct` only (3B/8B are commented out), and `run_sweep_fig5.py`'s `datasets` list is scoped to `nq_dataset.jsonl` only — both for faster iteration during development. Uncomment the additional models/datasets in each script to restore full paper-scale coverage; runtime scales roughly linearly with each.

## 2. Pre-Run Checklist

Before running, verify:
- `HF_TOKEN` is exported in your shell (open a new terminal if you only added it to `~/.bashrc`) and has accepted the Meta license for the gated Llama checkpoints used in the paper.
- The `ragmark:local` Docker image is built: `docker build -t ragmark:local .` from the repo root.
- `.venv` has `jupyter`, `pandas`, and `matplotlib` installed (see the main `README.md`'s First-run setup) — needed for the headless notebook execution step, which runs on the host, not inside the container.
- These paths in `config.py`:
  - `HF_HOME`
  - `BASE_OUT` (now a directory inside the repo — `bench_raw_data/` by default — not a separate data drive)
  - `WIKI_INDEX_DIR`
  - `WIKI_CORPUS_2018`
  - `DATASET_DIR`

Expected required files:
- FAISS index: `WIKI_INDEX_DIR/e5-base-v2-2018/index/e5_Flat.index`
- corpus JSONL: `WIKI_CORPUS_2018`
- dataset JSONL(s) under `DATASET_DIR` (one per benchmark used by the sweep configs)

## 3. What Outputs to Expect

Sweep outputs are written to:
- `BASE_OUT/testingevals{OUT_SUFFIX}/` — per-query accuracy CSVs
- `BASE_OUT/timingevals{OUT_SUFFIX}/` — per-query timing/performance JSONLs and CSVs
- `BASE_OUT/fig4_fig9_total_sweep_time.txt`, `BASE_OUT/fig5_total_sweep_time.txt` — total wall-clock time per sweep

With the default `config.py`, `BASE_OUT` is `bench_raw_data/` in the repo.

Per configuration, the artifact writes:
- `results_*.csv` (quality): `gold`, `prediction`, `rouge`, `f1`, `em`, `recall`
- `perfiii_*.csv` and `perfiii_*.jsonl` (performance): TTFT, decode/generation timing, tokens/sec, memory/power fields
- `trace_*.jsonl` (sampled stage traces): query/batch-level instrumentation events

After the notebook runs, `ae_evaluation/figures/plots/` holds the Fig. 4 / 5 / 9 reproduction PNGs and `ae_evaluation/figures/data/` holds the CSV tables each one was built from.

## 4. Running Representative Paper Experiments

`ae_eval.sh` runs both of these in sequence; each can also be run standalone (see Section 1):

```bash
python ae_evaluation/run_sweep_fig4_fig9.py
# or, for the compression-rate sweep:
python ae_evaluation/run_sweep_fig5.py
```

Notes:
- These scripts group runs to avoid reloading fixed components for every configuration.
- Existing completed configurations are skipped when expected output files already exist, so interrupted sweeps resume cleanly — rerun the same command.
- Runtime is written to `BASE_OUT/{fig4_fig9,fig5}_total_sweep_time.txt` after each run.

Paper-aligned sweep definitions (from the appendix):
- Exp. 1 / Fig. 4 (reranking latency/accuracy): `rerank=True`, `top_ks=[0,1,3,5,10]`, all 3 generators x 6 datasets.
- Exp. 2 / Fig. 5 (compression-rate latency): `compress=True`, `compress_method="llmlingua2"`, `top_ks=[1,3,5,10]`, `compress_rates=[0.2,0.4,0.6,0.8]`.
- Exp. 3 / Fig. 9 (TTFT-vs-ROUGE trade-off): naive configuration (`rerank=False`, `compress=False`), `top_ks=[0,1,3,5,10]`, all 3 generators.

These settings are configured in `run_sweep_fig4_fig9.py` (Fig. 4 / Fig. 9) or `run_sweep_fig5.py` (Fig. 5) and executed through `run_sweep.py` (repo root — imported via `sys.path` regardless of invocation directory).

## 5. Script-to-Figure/Table Data Mapping

Use this mapping when verifying how data is produced:

- `run_sweep_fig4_fig9.py` / `run_sweep_fig5.py` -> `run_sweep.py` -> `results_*.csv`, `perfiii_*.csv/jsonl`, `trace_*.jsonl`
  - Main source for multi-configuration comparison figures/tables.
- `load_evals_executed.ipynb`
  - Loads and aggregates `results_*`, `perfiii_*`, and `trace_*` files into `df_results`/`df_perf`/`df_trace` DataFrames.
  - Renders the Fig. 4 / 5 / 9 reproductions and writes each to `ae_evaluation/figures/plots/*.png` alongside the exact aggregated table it was built from in `ae_evaluation/figures/data/*.csv`.

Figure-specific analysis mapping:
- Fig. 4 (Case 2, reranking): combine `df_perf` stage latencies (embed, FAISS, TTFT, decode) plus `df_trace`'s `dur_rerank` (see Section 7 — rerank duration is not reliable in `df_perf`) with `df_results` ROUGE-L, grouped by dataset x model x top-k.
- Fig. 5 (Case 3, compression): filter compression sweep in `df_perf` (embed, FAISS, TTFT, decode) plus `df_trace`'s `dur_compress`, grouped by top-k x compression rate, for one model at a time (`MODEL_TO_PLOT` in the notebook cell).
- Fig. 9 (Case 1, naive): merge `df_results` (ROUGE-L) with `df_perf` (TTFT, decode, GPU0 peak memory, GPU0 mean SM%), grouped by model x top-k under naive pipeline settings.

## 6. Representative Runtime Expectations

Runtime depends heavily on `gen_models`/`datasets` scope (see Section 1's note on the checked-in grids), model size, GPU type, and index placement (CPU vs GPU). Rather than a fixed estimate, `ae_eval.sh` writes the actual measured total for each sweep to `BASE_OUT/{fig4_fig9,fig5}_total_sweep_time.txt` — check those after a run for this machine's real numbers.

## 7. Common Failure Modes

- `ModuleNotFoundError: No module named 'faiss'`, or FAISS retrieval silently running on CPU (`faiss_is_gpu=False` in `df_perf`)
  - The Docker image must install `faiss-gpu` (not `faiss-cpu`) for GPU-sharded retrieval — see the `conda install -c pytorch -c nvidia faiss-gpu=1.8.0` step in the `Dockerfile`. Rebuild if this was changed.
- `401`/gated-repo errors downloading Llama checkpoints
  - `HF_TOKEN` isn't set (or isn't exported in the shell that launched `docker run`/`ae_eval.sh`) or hasn't accepted the Meta license for the requested model.
- `ConnectionResetError` / `Connection aborted` loading the LLMLingua-2 compressor (Fig. 5 sweep)
  - `llmlingua`'s `PromptCompressor` unconditionally initializes a `tiktoken` GPT tokenizer, which fetches `cl100k_base.tiktoken` from `openaipublic.blob.core.windows.net` — often blocked on restricted networks even when `huggingface.co` is reachable. Fix: pre-seed a local tiktoken cache (the file is also mirrored on HF, e.g. `microsoft/Phi-3-small-8k-instruct`) and point `TIKTOKEN_CACHE_DIR` at it; `ae_eval.sh` warns if this directory is missing.
- `torch.OutOfMemoryError: CUDA out of memory` on GPU0 with larger models + reranking
  - `EMBED_DEVICE`/`GEN_DEVICE`/`COMPRESS_DEVICE`/`RERANK_DEVICE` all default to `cuda:0` in the sweep scripts, stacking on top of whatever FAISS index shard also lands there — this matches the paper's original A100-80GB layout but doesn't leave much headroom on smaller GPUs. Spread components across GPUs (e.g. `RERANK_DEVICE="cuda:1"`), or shard the generator itself across multiple GPUs by passing a comma-separated list to `--gen_device` (e.g. `"cuda:0,cuda:1"`), same convention as `--index_device`.
- CUDA unavailable
  - Set run devices to CPU through the device-config variables at the top of each sweep script (`EMBED_DEVICE`, `GEN_DEVICE`, `INDEX_DEVICE`), understanding runtime will increase substantially.

## 8. Reproducibility Notes

- `run_sweep.py` encodes run configuration in output filenames.
- `run_sweep.py` uses fixed random seed sampling (`random.seed(55)`) for evaluation subset selection.
- Re-running with identical inputs and environment should reproduce the same selected evaluation subset and output file naming convention.
