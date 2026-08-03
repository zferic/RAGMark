# RAGMark Artifact Evaluation Guide (IISWC 2026)

This guide is for artifact reviewers using SSH access to a prepared evaluation machine.

RAGMark reproduces the paper's five characterization cases: naive, reranking, compression, combined, and iterative.

Scope:
- No cloud setup.
- No container registry workflows.
- No large data download pipeline requirements.
- Use machine-local datasets, FAISS index files, and model cache already provisioned for evaluation.

## 1. Reviewer Quick Start

From the repository root:

```bash
bash scripts/ae_quick.sh
```

This command runs one reduced, representative benchmark configuration through the full RAGMark execution path:
- query embedding
- FAISS retrieval
- generation
- quality metrics export
- timing/performance export

By default, `ae_quick.sh` uses:
- model: `meta-llama/Llama-3.2-1B-Instruct`
- retriever: `intfloat/e5-base-v2`
- dataset: `nq_dataset.jsonl`
- eval size: 20 examples
- pipeline: `standard`
- index placement: `cpu`

You can override any of these through environment variables (documented in the script).

## 2. Pre-Run Checklist

Before running, verify these paths in `config.py`:
- `HF_HOME`
- `BASE_OUT`
- `WIKI_INDEX_DIR`
- `WIKI_CORPUS_2018`
- `DATASET_DIR`

Expected required files:
- FAISS index: `WIKI_INDEX_DIR/e5-base-v2-2018/index/e5_Flat.index`
- corpus JSONL: `WIKI_CORPUS_2018`
- dataset JSONL: `DATASET_DIR/nq_dataset.jsonl` (or selected dataset)
- HF access token with accepted Meta model licenses for gated Llama checkpoints used in the paper.

## 3. What Outputs to Expect

Quick run outputs are written to:
- `BASE_OUT/testingevals{AE_OUT_SUFFIX}`
- `BASE_OUT/timingevals{AE_OUT_SUFFIX}`

Per configuration, the artifact writes:
- `results_*.csv` (quality): `gold`, `prediction`, `rouge`, `f1`, `em`, `recall`
- `perfiii_*.csv` and `perfiii_*.jsonl` (performance): TTFT, decode/generation timing, tokens/sec, memory/power fields
- `trace_*.jsonl` (sampled stage traces): query/batch-level instrumentation events

## 4. Running Representative Paper Experiments

Primary experiment path used for paper-scale benchmarking:

```bash
python testing_callsmain_sweep.py
```

Notes:
- This script groups runs to avoid reloading fixed components for every configuration.
- Runtime is much longer than quick mode.
- Existing completed configurations are skipped when expected output files already exist.

Secondary validation path:

```bash
python testing_callsmain.py
```

Use this for independent single-config style runs via `main.py`.

Paper-aligned sweep definitions (from the appendix):
- Exp. 1 / Fig. 4 (reranking latency/accuracy): `rerank=True`, `top_ks=[0,1,3,5,10]`, all 3 generators x 6 datasets.
- Exp. 2 / Fig. 5 (compression-rate latency): `compress=True`, `compress_method="llmlingua2"`, `top_ks=[1,3,5,10]`, `compress_rates=[0.2,0.4,0.6,0.8]`.
- Exp. 3 / Fig. 9 (TTFT-vs-ROUGE trade-off): naive configuration (`rerank=False`, `compress=False`), `top_ks=[0,1,3,5,10]`, all 3 generators.

These settings are configured in `testing_callsmain_sweep.py` and executed through `main_sweep.py`.

## 5. Script-to-Figure/Table Data Mapping

Use this mapping when verifying how data is produced:

- `testing_callsmain_sweep.py` -> `main_sweep.py` -> `results_*.csv`, `perfiii_*.csv/jsonl`, `trace_*.jsonl`
  - Main source for multi-configuration comparison figures/tables.
- `testing_callsmain.py` -> `main.py` -> `results_*.csv`, `perfiii_*.csv/jsonl`
  - Cross-check path for per-configuration comparisons.
- `load_evals_executed.ipynb`
  - Loads and aggregates `results_*`, `perfiii_*`, and `trace_*` files into analysis DataFrames used for plotting/reporting.

Figure-specific analysis mapping:
- Fig. 4: combine `df_perf` stage latencies (embed, FAISS, rerank, TTFT, decode) with `df_results` ROUGE-L, grouped by dataset x model x top-k.
- Fig. 5: filter compression sweep in `df_perf`, grouped by top-k x compression rate; plot TTFT/decode/compress/FAISS/embed latency.
- Fig. 9: merge `df_results` (ROUGE-L) with `df_perf` (TTFT, decode, memory, SM utilization), grouped by model x top-k under naive pipeline settings.

## 6. Representative Runtime Expectations

Approximate expectations on a prepared GPU machine:
- Quick check (`bash scripts/ae_quick.sh`): usually minutes.
- Full sweep (`python testing_callsmain_sweep.py` with multiple models/datasets): hours.

Runtime variability is expected based on model size, GPU type, and index placement (CPU vs GPU).

## 7. Common Failure Modes

- `ModuleNotFoundError: No module named 'faiss'`
  - Install a FAISS build compatible with the machine environment (`faiss-gpu` or `faiss-cpu`).
- Path/file not found in quick script checks
  - Confirm `config.py` points to provisioned dataset/index/corpus locations.
- CUDA unavailable
  - Set run devices to CPU through script overrides (`AE_EMBED_DEVICE=cpu`, `AE_GEN_DEVICE=cpu`, `AE_INDEX_DEVICE=cpu`), understanding runtime will increase.

## 8. Reproducibility Notes

- `main_sweep.py` and `main.py` encode run configuration in output filenames.
- `main_sweep.py` uses fixed random seed sampling (`random.seed(55)`) for evaluation subset selection.
- Re-running with identical inputs and environment should reproduce the same selected evaluation subset and output file naming convention.
