# RAGMark Artifact Evaluation Guide (IISWC 2026)

RAGMark reproduces the paper's five characterization cases: naive, reranking, compression, combined, and iterative.

This artifact received the Available, Reviewed, and Reproducible badges. The evaluated version is
archived at [doi.org/10.5281/zenodo.22058207](https://doi.org/10.5281/zenodo.22058207); `main` may
have changed since.

- **Time:** ~1 hour for all AE experiments and figures
- **Disk:** ~65 GB for data and code (FAISS index, cached model weights, and the Wikipedia corpus
  dominate), plus ~10 GB for the Docker image

> **Note on hardware:** the paper's results (Section 4.1) were measured on two NVIDIA A100-SXM4-40GB
> GPUs, which the university is unable to provide remote access to for artifact evaluation. We
> therefore provide this evaluation machine instead, equipped with two NVIDIA V100-PCIE-16GB
> GPUs — an older generation with less memory per device. Absolute numbers (latency, energy,
> memory) will not match the paper on this older-generation hardware. This artifact reproduces
> several of the figures and datasets, and the methodology behind them — not the paper's exact
> reported values. Those can, however, be easily reproduced on a machine matching the
> specifications in the paper. We believe this also showcases the portability of RAGMark across
> hardware, and the significant impact that both the RAG configuration and the underlying hardware
> platform have on the final results.

## 1. Access (used during evaluation)

During artifact evaluation, reviewers SSH'd into a prepared evaluation machine (credentials were
provided separately):

```bash
ssh <username>@129.10.225.5
```

## 2. Run Everything

From the `RAGMark` repo root:

```bash
bash ae_evaluation/ae_eval.sh
```

This runs both sweeps and the analysis notebook end-to-end, and prints where the results land.

## 3. Where Results Land

- `RAGMark/bench_raw_data/testingevals_local/` — per-query accuracy CSVs
- `RAGMark/bench_raw_data/timingevals_local/` — per-query timing/performance data
- `RAGMark/bench_raw_data/fig4_fig9_total_sweep_time.txt`, `fig5_total_sweep_time.txt` — total runtime per sweep
- `RAGMark/ae_evaluation/figures/plots/` — Fig. 4 / 5 / 9 reproductions (PNG)
- `RAGMark/ae_evaluation/figures/data/` — the data table behind each figure (CSV)

To regenerate the figures without opening Jupyter, execute the notebook headlessly:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace ae_evaluation/load_evals_executed.ipynb
```

This runs every cell and saves the outputs back into the `.ipynb`, so reopening it later still shows
the results. It requires `jupyter`, `pandas`, and `matplotlib` in a `.venv` at the repo root
(`python3 -m venv .venv && .venv/bin/pip install jupyter pandas matplotlib`).

## 4. Running Experiments Individually

`ae_eval.sh` runs both of these; either can also be run on its own:

```bash
python ae_evaluation/run_sweep_fig4_fig9.py
python ae_evaluation/run_sweep_fig5.py
```

Paper-aligned sweep definitions (from the appendix):
- Exp. 1 / Fig. 4 (reranking latency/accuracy): `rerank=True`, `top_ks=[0,1,3,5,10]`, all 3 generators x 6 datasets.
- Exp. 2 / Fig. 5 (compression-rate latency): `compress=True`, `compress_method="llmlingua2"`, `top_ks=[1,3,5,10]`, `compress_rates=[0.2,0.4,0.6,0.8]`.
- Exp. 3 / Fig. 9 (TTFT-vs-ROUGE trade-off): naive configuration (`rerank=False`, `compress=False`), `top_ks=[0,1,3,5,10]`, all 3 generators.
