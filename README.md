# RAGMark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RAGMark is a benchmarking pipeline for Retrieval-Augmented Generation (RAG) systems. It evaluates quality and performance together, including retrieval recall, answer accuracy, latency, TTFT/decode time, and GPU behavior.

The paper characterizes five RAG cases: naive, reranking, compression, combined, and iterative.

For IISWC artifact evaluation, start with [AE_README.md](AE_README.md).

## Quick Start (Prepared Evaluation Machine)

This project assumes the evaluation machine already has datasets, FAISS index files, model cache, and Python dependencies.

Reviewers also need HuggingFace access for gated Meta Llama models (via HF token and accepted model licenses).

1. Check paths in `config.py`.
2. Run a representative end-to-end benchmark:

```bash
bash scripts/ae_quick.sh
```

3. Inspect outputs in:

- `BASE_OUT/testingevals{OUT_SUFFIX or AE override}`
- `BASE_OUT/timingevals{OUT_SUFFIX or AE override}`

The quick script executes embedding, FAISS retrieval, optional reranking/compression settings (disabled by default in quick mode), generation, and metric/timing output export.

## Repository Overview

### Core entry points

- `main.py`: single-configuration execution path.
- `main_sweep.py`: grouped sweep execution path (loads fixed components once per group).
- `testing_callsmain.py`: grid orchestrator for `main.py`.
- `testing_callsmain_sweep.py`: sweep orchestrator for `main_sweep.py`.

### Configuration and analysis

- `config.py`: machine-local paths for model cache, datasets, FAISS index, and output directories.
- `load_evals_executed.ipynb`: aggregates output files into analysis DataFrames.

### Pipeline modules

- `rag_system/embedding.py`: query embedding.
- `rag_system/retrieval.py`: FAISS retrieval.
- `rag_system/reranking.py`: cross-encoder reranking.
- `rag_system/compression.py`: context compression methods.
- `rag_system/generation.py`: answer generation.
- `rag_system/pipelines/standard.py`: single-pass RAG pipeline.
- `rag_system/pipelines/iterative.py`: iterative retrieval-generation pipeline.
- `rag_system/timing.py`: timing, memory, and trace instrumentation.

## Output Files

For each run configuration, RAGMark writes:

- `results_*.csv` in `testingevals*`: per-query quality outputs and metrics.
  - Columns: `gold`, `prediction`, `rouge`, `f1`, `em`, `recall`.
- `perfiii_*.jsonl` and `perfiii_*.csv` in `timingevals*`: latency and system-level performance traces.
- `trace_*.jsonl` in `timingevals*`: fine-grained sampled trace events for selected queries/batches.

Completed configurations are skipped automatically on re-run when all expected output files are present.

## Script-to-Paper Mapping

Use this mapping to connect artifact scripts to paper figures/tables:

- Fig. 4 (reranking latency/accuracy): `testing_callsmain_sweep.py` with reranking enabled (`rerank=True`) and top-k sweep, analyzed via `load_evals_executed.ipynb` by combining `df_perf` stage latencies with `df_results` ROUGE-L.
- Fig. 5 (compression-rate latency): `testing_callsmain_sweep.py` with compression enabled (`compress=True`, `compress_method="llmlingua2"`) and `compress_rates=[0.2, 0.4, 0.6, 0.8]`, analyzed from `df_perf` grouped by top-k and compression rate.
- Fig. 9 (TTFT vs ROUGE trade-off): `testing_callsmain_sweep.py` in naive mode (`rerank=False`, `compress=False`) with top-k sweep, analyzed by merging `df_results` ROUGE-L with `df_perf` TTFT/decode and utilization fields.
- `testing_callsmain.py` + `main.py`: secondary path for independent per-configuration validation.
- `trace_*.jsonl` plus timing fields in `rag_system/timing.py`: stage-level overhead breakdown (embedding, retrieval, reranking, compression, generation).

## Representative Runtime Guidance

Actual runtime depends on hardware, model size, and index placement. On a prepared GPU machine:

- `scripts/ae_quick.sh` (1 dataset, reduced eval size): typically minutes.
- `testing_callsmain_sweep.py` default full set: typically hours (multiple models and datasets).

The quick script is intended for reviewer sanity-check and pipeline validation, not for reproducing full paper totals.

## Minimal Setup Notes

`requirements.txt` contains Python package requirements. FAISS is intentionally not pinned there because most evaluation environments install FAISS through conda (`faiss-gpu` or `faiss-cpu`) to match machine CUDA/CPU configuration.

If you see `ModuleNotFoundError: No module named 'faiss'`, install a FAISS build compatible with the evaluation environment.
