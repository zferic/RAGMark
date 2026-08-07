# RAGMark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An experimental benchmark for Retrieval-Augmented Generation (RAG) systems across models, datasets, retrieval indexes, pipelines, and compression settings, recording accuracy and performance traces per query.

## Features

- Per-query, per-stage profiling: embedding, FAISS retrieval, reranking, compression, generation
- Accuracy metrics (EM, F1, ROUGE-L, retrieval recall) and performance metrics (TTFT, decode latency, tokens/sec, GPU power/energy, utilization)
- Loads models and indexes once per sweep group to avoid redundant reloading across configs
- Standard (single-pass) and iterative (multi-round retrieval) pipelines
- Independent CPU/GPU device placement per component (embedder, index, reranker, compressor, generator)
- Skips already-completed configs, so interrupted sweeps resume cleanly

## Models & datasets evaluated in the publication

| | |
|---|---|
| **Generators** | Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3-8B-Instruct |
| **Embedders** | e5-small-v2, e5-base-v2, e5-large-v2 |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Compression** | LLMLingua, LLMLingua-2, Selective-Context |
| **Index types** | Flat, IVF, IVF-SQ |
| **QA datasets** | NQ, TriviaQA, SQuAD, WebQuestions, PopQA, HotpotQA (via [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)) |
| **Retrieval corpus** | Wikipedia 2018 dump (~9.2M passages) |

This table reflects what the paper's experiments cover, not a hard limit — the framework is built around standard Hugging Face interfaces, so swapping in pretty much any other Hugging Face generator, embedder, reranker, or dataset is a config change, not a code change.

## Artifact Evaluation (IISWC 2026)

For reviewers: all AE documentation, experiment definitions, and the end-to-end runner are in [`ae_evaluation/`](ae_evaluation/).

```bash
# Run both sweeps + the analysis notebook end-to-end via Docker:
bash ae_evaluation/ae_eval.sh

# Or run either sweep individually:
python ae_evaluation/run_sweep_fig4_fig9.py
# or, for the compression-rate sweep:
python ae_evaluation/run_sweep_fig5.py
```

See [`ae_evaluation/AE_README.md`](ae_evaluation/AE_README.md) for reviewer access and where results land.

## First-run setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

FAISS must be installed separately via conda (the pip package does not include GPU support):

```bash
conda install -c pytorch faiss-gpu=1.8.0
# or for CPU-only:
conda install -c pytorch faiss-cpu
```

### 2. Edit `config.py`

All paths are centralized in `config.py`. Edit this file before running anything:

```python
HF_HOME = "/path/to/huggingface/cache"   # where HF models are downloaded/cached

BASE_OUT   = "/path/to/output/dir"        # where results CSVs and trace files are written
OUT_SUFFIX = "_a100"                      # appended to output subdirs (e.g. testingevals_a100)

WIKI_INDEX_DIR   = "/path/to/wikiindex/"  # root dir containing FAISS index subdirs
WIKI_CORPUS_2018 = "/path/to/test_sample_2018_sent.jsonl"  # Wikipedia 2018 corpus JSONL

DATASET_DIR = "/path/to/flashrag/dataset" # dir containing *_dataset.jsonl eval files
```

Output is written to:
- `BASE_OUT/testingevals{OUT_SUFFIX}/` — per-query accuracy CSVs
- `BASE_OUT/timingevals{OUT_SUFFIX}/` — per-query timing/performance JSONLs and CSVs

### 3. Prepare data

**FAISS index** — each retrieval model needs a prebuilt index under `WIKI_INDEX_DIR`:
```
WIKI_INDEX_DIR/
  e5-base-v2-2018/index/e5_Flat.index
  e5-small-v2-2018/index/e5_Flat.index
  ...
```

**Corpus** — a JSONL file at `WIKI_CORPUS_2018` with one passage per line.

**Eval datasets** — JSONL files in `DATASET_DIR`, one per benchmark (hotpotqa, nq, squad, triviaqa, popqa, webquestions). These are available from [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG).

### 4. Configure the sweep

`run_sweep.py` is the general execution engine — it takes a JSON-encoded list of configs
(`--sweep_configs`) plus model/retriever/device args, evaluates each one, and writes results,
grouping runs by model+retriever+dataset so fixed components load only once. In practice you
drive it via a small orchestrator script that builds that JSON and invokes `run_sweep.py` as a
subprocess. `ae_evaluation/run_sweep_fig4_fig9.py` and `ae_evaluation/run_sweep_fig5.py` (example
configurations, wired up for this paper's own experiments) are a good starting point to copy and
adapt:

- `gen_models` — list of HuggingFace generator model IDs
- `ret_models` — list of `[retrieval_model_id, index_path, corpus_jsonl]` triples
- `datasets` — subset of eval datasets to run (pulled from `DATASET_DIR` via config)
- `userags`, `top_ks`, `batches`, `compresss` — sweep dimensions
- `EMBED_DEVICE`, `GEN_DEVICE`, `INDEX_DEVICE` — device placement

### 5. Run

```bash
python your_sweep_script.py   # e.g. a copy of one of the example configurations above
```

Results are skipped automatically if output files already exist.

### 6. Load results

Open `ae_evaluation/load_evals_executed.ipynb` in Jupyter. It reads all output files from `testingevals{OUT_SUFFIX}/` and `timingevals{OUT_SUFFIX}/` and builds three DataFrames:

- `df_results` — per-query accuracy metrics
- `df_perf` — per-query timing and GPU performance
- `df_trace` — per-query stage-level timing traces and system sensor data

Config fields (model, dataset, use_rag, topk, etc.) are parsed from filenames into columns automatically.

Update the `BASE_OUT` and `OUT_SUFFIX` variables at the top of the notebook to match your `config.py`.

**Running headlessly (no notebook UI):** to execute every cell and produce the Figure 4 / 5 / 9
reproductions (PNGs + CSVs under `ae_evaluation/figures/{plots,data}/`) without opening Jupyter:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace ae_evaluation/load_evals_executed.ipynb
```

This runs all cells and saves the outputs back into the `.ipynb` in place, so reopening it later still
shows the results. Requires `jupyter`, `pandas`, and `matplotlib` installed in `.venv` (see First-run setup).

## High Level Repository Overview

| File | Purpose |
|------|---------|
| `config.py` | All configurable paths — edit this first |
| `run_sweep.py` | Multi-config execution engine; loads fixed components once per sweep group |
| `ae_evaluation/run_sweep_fig4_fig9.py` | Sweep orchestrator for Fig. 4 / Fig. 9 (reranking and naive runs) |
| `ae_evaluation/run_sweep_fig5.py` | Sweep orchestrator for Fig. 5 (compression-rate sweep) |
| `ae_evaluation/load_evals_executed.ipynb` | Loads and merges all result files into DataFrames |
| `ae_evaluation/ae_eval.sh` | Runs both sweeps + the analysis notebook end-to-end via Docker |
| `requirements.txt` | Python dependencies |
| `rag_system/` | Core RAG modules (embedding, retrieval, generation, compression, reranking, timing) |
