"""
testing_callsmain_sweep.py

Orchestrator for main_sweep.py.
Groups configs by (model, ret_model, dataset) so that embedder/FAISS/generator
are loaded ONCE per group instead of once per config.
"""
import subprocess
import os
import sys
import json

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
from config import BASE_OUT, OUT_SUFFIX, base_index_data, data_index2018, DATASET_DIR

EVAL_SIZE  = 100
BATCH_SIZE = 1

gen_models = [
    #"meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]

ret_models = [
     ["intfloat/e5-base-v2",
      base_index_data + "e5-base-v2-2018/index/e5_Flat.index",
      data_index2018],
]


'''
base_index_data = "/projects/nucar/feric.z/wikiindex2/test_sample_2018_sent"
data_index2018  = "/projects/nucar/feric.z/FLashRAG-z/test_sample_2018_sent.jsonl"

def _discover_indexes(base_dir):
    """Return [ret_model, index_path, corpus_path] for every index subdir found."""
    import glob
    entries = []
    for subdir in sorted(glob.glob(os.path.join(base_dir, "index_IVF*"))):
        # e.g. index_IVF1000-Flat_ml256  ->  e5_IVF1000,Flat.index
        name = os.path.basename(subdir)          # index_IVF1000-Flat_ml256
        core = name.replace("index_", "", 1)     # IVF1000-Flat_ml256
        core = core[:core.rfind("_ml")]          # IVF1000-Flat
        idx_name = "e5_" + core.replace("-", ",") + ".index"
        idx_path = os.path.join(subdir, idx_name)
        entries.append(["intfloat/e5-small-v2", idx_path, data_index2018])
    if not entries:
        raise RuntimeError(f"No index subdirs found under {base_dir}")
    return entries

ret_models = _discover_indexes(base_index_data)
'''

datasets = [
    os.path.join(DATASET_DIR, "hotpotqa_dataset.jsonl"),
    os.path.join(DATASET_DIR, "nq_dataset.jsonl"),
    os.path.join(DATASET_DIR, "squad_dataset.jsonl"),
    os.path.join(DATASET_DIR, "triviaqa_dataset.jsonl"),
    os.path.join(DATASET_DIR, "popqa_dataset.jsonl"),
    os.path.join(DATASET_DIR, "webquestions_dataset.jsonl"),
]

#pipelines = ["iterative"]
pipelines = ["standard"]

reranks        = [False]
rerank_top_ns  = [3, 5]
rerank_models  = ["cross-encoder/ms-marco-MiniLM-L-6-v2"]

# Device config
EMBED_DEVICE    = "cuda:0"
GEN_DEVICE      = "cuda:0"
INDEX_DEVICE    = "cuda:0"   # change to "cpu" for hardware placement run
#INDEX_DEVICE    = "cuda:0"   # change to "cpu" for hardware placement run
INDEX_DEVICE    = "cpu"   # change to "cpu" for hardware placement run

COMPRESS_DEVICE = "cuda:0"
RERANK_DEVICE   = "cuda:0"

# Inner sweep parameters
userags          = [False, True]
compresss        = [True]
compress_methods = ["llmlingua2", "sc"]   # longllmlingua excluded
compress_rates   = [.25, 0.5, .75]
top_ks           = [1,3,5,10]
batches          = [False, True]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

# -------------------------------------------------------------------------
# Build sweep configs for a given (use_rag) combination
# -------------------------------------------------------------------------
def build_sweep_configs():
    configs = []
    for use_rag in userags:
        run_top_ks = top_ks if use_rag else [0]
        for top_k in run_top_ks:
            for compress in (compresss if use_rag else [False]):
                if compress:
                    for method in compress_methods:
                        for rate in compress_rates:
                            configs.append({
                                "use_rag": use_rag,
                                "compress": True,
                                "compress_method": method,
                                "compression_rate": rate,
                                "rag_top_k": top_k,
                            })
                else:
                    configs.append({
                        "use_rag": use_rag,
                        "compress": False,
                        "compress_method": "none",
                        "compression_rate": 1.0,
                        "rag_top_k": top_k,
                    })
    return configs

# -------------------------------------------------------------------------
# Run one sweep (one subprocess = one model × ret, loops over all datasets)
# -------------------------------------------------------------------------
def run_sweep(model, ret_model, index_path, corpus_path, eval_paths, batch=True,
              rerank=False, rerank_model=rerank_models[0], rerank_top_n=3, pipeline="standard"):
    sweep_configs = build_sweep_configs()
    sweep_json    = json.dumps(sweep_configs)
    paths_json    = json.dumps(eval_paths)

    command = [
        "python", "-u", "main_sweep.py",
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
    if rerank:
        command.extend(["--rerank", "--rerank_model", rerank_model, "--rerank_top_n", str(rerank_top_n)])

    print("\n" + "=" * 80)
    print(f"MODEL:    {model}")
    print(f"RET:      {ret_model}")
    print(f"RERANK:   {rerank} model={rerank_model} top_n={rerank_top_n}")
    print(f"DEVICES:  embed={EMBED_DEVICE} gen={GEN_DEVICE} index={INDEX_DEVICE} compress={COMPRESS_DEVICE} rerank={RERANK_DEVICE}")
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
# Main grid — one subprocess per (model × ret_model), all datasets inside
# -------------------------------------------------------------------------
for model in gen_models:
    for ret_model, index_path, corpus_path in ret_models:
        for pipeline in pipelines:
            for batch in batches:
                for rerank in reranks:
                    for rerank_top_n in (rerank_top_ns if rerank else [0]):
                        for rerank_model in (rerank_models if rerank else [rerank_models[0]]):
                            rc = run_sweep(model, ret_model, index_path, corpus_path, datasets,
                                           batch=batch, rerank=rerank, rerank_model=rerank_model,
                                           rerank_top_n=rerank_top_n, pipeline=pipeline)
                            if rc != 0:
                                print(f"FAILED: Sweep failed (rc={rc}) for {model} / {ret_model} / "
                                      f"pipeline={pipeline} / batch={batch} / rerank={rerank} / {rerank_model}")
