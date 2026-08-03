# RAGMark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A benchmarking pipeline for Retrieval-Augmented Generation (RAG) systems. Runs a configurable sweep of models, datasets, retrieval indexes, and compression settings, recording accuracy and performance traces per query.

## Features

- Per-query, per-stage profiling: embedding, FAISS retrieval, reranking, compression, generation
- Accuracy metrics (EM, F1, ROUGE-L, retrieval recall) and performance metrics (TTFT, decode latency, tokens/sec, GPU power/energy, utilization)
- Loads models and indexes once per sweep group to avoid redundant reloading across configs
- Standard (single-pass) and iterative (multi-round retrieval) pipelines
- Independent CPU/GPU device placement per component (embedder, index, reranker, compressor, generator)
- Skips already-completed configs, so interrupted sweeps resume cleanly

## Supported models & datasets

| | |
|---|---|
| **Generators** | Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3-8B-Instruct |
| **Embedders** | e5-small-v2, e5-base-v2, e5-large-v2 |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Compression** | LLMLingua, LLMLingua-2, Selective-Context |
| **Index types** | Flat, IVF, IVF-SQ |
| **QA datasets** | NQ, TriviaQA, SQuAD, WebQuestions, PopQA, HotpotQA (via [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)) |
| **Retrieval corpus** | Wikipedia 2018 dump (~9.2M passages) |

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

Edit `run_sweep_configs.py` to set:

- `gen_models` — list of HuggingFace generator model IDs
- `ret_models` — list of `[retrieval_model_id, index_path, corpus_jsonl]` triples
- `datasets` — subset of eval datasets to run (pulled from `DATASET_DIR` via config)
- `userags`, `top_ks`, `batches`, `compresss` — sweep dimensions
- `EMBED_DEVICE`, `GEN_DEVICE`, `INDEX_DEVICE` — device placement

### 5. Run

```bash
# Sweep mode (groups configs by model+retriever+dataset, loads each once):
python run_sweep_configs.py
```

Results are skipped automatically if output files already exist.

### 6. Load results

Open `load_evals_executed.ipynb` in Jupyter. It reads all output files from `testingevals{OUT_SUFFIX}/` and `timingevals{OUT_SUFFIX}/` and builds three DataFrames:

- `df_results` — per-query accuracy metrics
- `df_perf` — per-query timing and GPU performance
- `df_trace` — per-query stage-level timing traces and system sensor data

Config fields (model, dataset, use_rag, topk, etc.) are parsed from filenames into columns automatically.

Update the `BASE_OUT` and `OUT_SUFFIX` variables at the top of the notebook to match your `config.py`.

## File overview

| File | Purpose |
|------|---------|
| `config.py` | All configurable paths — edit this first |
| `run_sweep.py` | Multi-config execution engine; loads fixed components once per sweep group |
| `run_sweep_configs.py` | Sweep orchestrator — prepares sweep settings and invokes `run_sweep.py` |
| `load_evals_executed.ipynb` | Loads and merges all result files into DataFrames |
| `scripts/ae_quick.sh` | Reduced representative AE run using sweep pipeline |
| `requirements.txt` | Python dependencies |
| `rag_system/` | Core RAG modules (embedding, retrieval, generation, compression, reranking, timing) |
