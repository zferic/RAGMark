# RAGMark Artifact Evaluation Guide (IISWC 2026)

RAGMark reproduces the paper's five characterization cases: naive, reranking, compression, combined, and iterative.

## 1. Access

SSH into the evaluation machine (credentials provided separately):

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
