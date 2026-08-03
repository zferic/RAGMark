"""
run_sweep_fig5.py

Orchestrator for Fig. 5 (compression-rate latency sweep).

Exp. 2 -- compress=True, compress_method="llmlingua2",
           top_ks=[1,3,5,10], compress_rates=[0.2,0.4,0.6,0.8],
           all 3 generators x 6 datasets.

Run via: python run_sweep_fig5.py
"""
import subprocess
import os
import sys
import json

from config import BASE_OUT, OUT_SUFFIX, base_index_data, data_index2018, DATASET_DIR

EVAL_SIZE  = 400
BATCH_SIZE = 1

gen_models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]

ret_models = [
    ["intfloat/e5-base-v2",
     os.path.join(base_index_data, "e5-base-v2-2018", "index", "e5_Flat.index"),
     data_index2018],
]

datasets = [
    os.path.join(DATASET_DIR, "hotpotqa_dataset.jsonl"),
    os.path.join(DATASET_DIR, "nq_dataset.jsonl"),
    os.path.join(DATASET_DIR, "squad_dataset.jsonl"),
    os.path.join(DATASET_DIR, "triviaqa_dataset.jsonl"),
    os.path.join(DATASET_DIR, "popqa_dataset.jsonl"),
    os.path.join(DATASET_DIR, "webquestions_dataset.jsonl"),
]

pipelines = ["standard"]

# Fig. 5: no reranking
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Device config
EMBED_DEVICE    = "cuda:0"
GEN_DEVICE      = "cuda:0"
INDEX_DEVICE    = "cpu"
COMPRESS_DEVICE = "cuda:0"
RERANK_DEVICE   = "cuda:0"

# Fig. 5 sweep parameters (per appendix Exp. 2)
top_ks           = [1, 3, 5, 10]
compress_rates   = [0.2, 0.4, 0.6, 0.8]
compress_method  = "llmlingua2"
batches          = [False]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

# -------------------------------------------------------------------------
# Build sweep configs: RAG + compression at each (top_k, rate) combination
# -------------------------------------------------------------------------
def build_sweep_configs():
    configs = []
    for top_k in top_ks:
        for rate in compress_rates:
            configs.append({
                "use_rag":          True,
                "compress":         True,
                "compress_method":  compress_method,
                "compression_rate": rate,
                "rag_top_k":        top_k,
            })
    return configs

# -------------------------------------------------------------------------
# Run one sweep (one subprocess = one model × ret, loops over all datasets)
# -------------------------------------------------------------------------
def run_sweep(model, ret_model, index_path, corpus_path, eval_paths, batch=False, pipeline="standard"):
    sweep_configs = build_sweep_configs()
    sweep_json    = json.dumps(sweep_configs)
    paths_json    = json.dumps(eval_paths)

    command = [
        "python", "-u", "run_sweep.py",
        "--model_path",      model,
        "--retrieval_model", ret_model,
        "--retrieval_index", index_path,
        "--retrieval_json",  corpus_path,
        "--eval_paths",      paths_json,
        "--eval_size",       str(EVAL_SIZE),
        "--batch_size",      str(BATCH_SIZE),
        *(["--batch"] if batch else []),
        "--base_out",        BASE_OUT,
        "--out_suffix",      OUT_SUFFIX,
        "--sweep_configs",   sweep_json,
        "--pipeline",        pipeline,
        "--embed_device",    EMBED_DEVICE,
        "--gen_device",      GEN_DEVICE,
        "--index_device",    INDEX_DEVICE,
        "--compress_device", COMPRESS_DEVICE,
        "--rerank_device",   RERANK_DEVICE,
    ]

    print("\n" + "=" * 80)
    print(f"MODEL:    {model}")
    print(f"RET:      {ret_model}")
    print(f"COMPRESS: method={compress_method} rates={compress_rates} top_ks={top_ks}")
    print(f"DEVICES:  embed={EMBED_DEVICE} gen={GEN_DEVICE} index={INDEX_DEVICE} compress={COMPRESS_DEVICE}")
    print(f"DATASETS: {len(eval_paths)} ({', '.join(os.path.basename(p) for p in eval_paths)})")
    print(f"CONFIGS:  {len(sweep_configs)} per dataset")
    print("=" * 80)

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            sys.stdout.flush()
        process.wait()
        return process.returncode
    except Exception as e:
        print(f"Error: {e}")
        return 1

# -------------------------------------------------------------------------
# Main grid
# -------------------------------------------------------------------------
for model in gen_models:
    for ret_model, index_path, corpus_path in ret_models:
        for pipeline in pipelines:
            for batch in batches:
                rc = run_sweep(model, ret_model, index_path, corpus_path, datasets,
                               batch=batch, pipeline=pipeline)
                if rc != 0:
                    print(f"FAILED: Sweep failed (rc={rc}) for {model} / {ret_model} / pipeline={pipeline}")
