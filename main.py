import argparse
import json

import os
import torch
import re
import random

from config import HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

# Keep auth vars; reset only cache paths
for k in [
    "TRANSFORMERS_CACHE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "XDG_CACHE_HOME",
    "HF_ASSETS_CACHE",
    "HF_DATASETS_CACHE",
    "HF_MODULES_CACHE",
]:
    os.environ.pop(k, None)

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_ASSETS_CACHE"] = os.path.join(HF_HOME, "assets")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["HF_MODULES_CACHE"] = os.path.join(HF_HOME, "modules")

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token

print("HF_HOME:", os.environ["HF_HOME"])
print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])
print("Token present:", "HF_TOKEN" in os.environ)



# -----------------------------------------------------------------------------
# Parse args
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, required=True)

parser.add_argument("--retrieval_model", type=str, required=True)
parser.add_argument("--retrieval_index", type=str, required=True)
parser.add_argument("--retrieval_json", type=str, required=True)

parser.add_argument("--eval_path", type=str, required=True)
parser.add_argument("--eval_size", type=int, default=1000)

#parser.add_argument("--use_rag", type=str, choices=["Yes", "No"], default="Yes")
parser.add_argument("--use_rag", action="store_true")
#parser.set_defaults(use_rag=True)
parser.add_argument("--rag_top_k", type=int, default=5)

#parser.add_argument("--compress", type=str, choices=["Yes", "No"], default="No")
parser.add_argument("--compress", action="store_true")
parser.add_argument("--compress_method", type=str, default="llmlingua2")
parser.add_argument("--compression_rate", type=float, default=0.5)

#parser.add_argument("--batch", type=str, choices=["Yes", "No"], default="No")
parser.add_argument("--batch", action="store_true")
parser.add_argument("--batch_size", type=int, default=8)


parser.add_argument("--pipeline", type=str, default="standard",
                    choices=["standard", "iterative"])
parser.add_argument("--iter_num", type=int, default=3)

parser.add_argument("--rerank", action="store_true")
parser.add_argument("--rerank_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
parser.add_argument("--rerank_top_n", type=int, default=3)
parser.add_argument("--rerank_device", type=str, default="cuda:0")

parser.add_argument("--embed_device", type=str, default="cuda:0")
parser.add_argument("--index_device", type=str, default="cuda:0,cuda:1")      # FAISS GPU (or "cpu")
parser.add_argument("--compress_device", type=str, default="cuda:0")
parser.add_argument("--gen_device", type=str, default="cuda:0")

args = parser.parse_args()


# -------------------------------------------------------------------------
# BASE OUTPUT DIRECTORY
# -------------------------------------------------------------------------
from config import BASE_OUT, OUT_SUFFIX

TESTING_DIR = f"{BASE_OUT}/testingevals{OUT_SUFFIX}"
TIMING_DIR  = f"{BASE_OUT}/timingevals{OUT_SUFFIX}"

os.makedirs(TESTING_DIR, exist_ok=True)
os.makedirs(TIMING_DIR, exist_ok=True)


def parse_device(s: str) -> torch.device:
    s = str(s)
    if s.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(s)

EMBED_DEVICE   = parse_device(args.embed_device)
COMPRESS_DEVICE= parse_device(args.compress_device)
GEN_DEVICE     = parse_device(args.gen_device)


# FAISS wants a list of int gpu ids, or [-1] for CPU
_index_parts = [s.strip() for s in args.index_device.split(",")]

if _index_parts == ["cpu"]:
    INDEX_DEVICE  = torch.device("cpu")
    INDEX_GPU_IDS = [-1]
else:
    INDEX_GPU_IDS = [torch.device(p).index or 0 for p in _index_parts]
    INDEX_DEVICE  = torch.device(_index_parts[0])  # first GPU, for embedding placement



print("EMBED_DEVICE:", EMBED_DEVICE)
print("INDEX_DEVICE:", INDEX_DEVICE, "INDEX_GPU_IDS:", INDEX_GPU_IDS)
print("COMPRESS_DEVICE:", COMPRESS_DEVICE)
print("GEN_DEVICE:", GEN_DEVICE)


def _safe(s: str, max_len: int = 180) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace(":", "-").replace("/", "-")
    s = re.sub(r"[^A-Za-z0-9._=-]", "-", s)
    return s[:max_len].strip("._-")

def device_tag(embed_device, index_device, index_gpu_ids, compress_device, gen_device) -> str:
    return _safe(
        f"dev"
        f"_embed={embed_device}"
        f"_index={index_device}"
        f"_indexgpu={'-'.join(str(i) for i in index_gpu_ids)}"
        f"_compress={compress_device}"
        f"_gen={gen_device}"
    )

import hashlib as _hashlib

dev_tag_full = device_tag(EMBED_DEVICE, INDEX_DEVICE, INDEX_GPU_IDS, COMPRESS_DEVICE, GEN_DEVICE)
dev_tag = "dev" + _hashlib.md5(dev_tag_full.encode()).hexdigest()[:6]

ret_name = os.path.basename(args.retrieval_model)          # e.g. "e5-large-v2"
idx_name = os.path.basename(args.retrieval_index).replace(".", "_")
corpus_name = os.path.basename(args.retrieval_json).replace(".jsonl", "")
model_name  = os.path.basename(args.model_path)            # e.g. "Llama-3.2-1B-Instruct"
dataset_name = os.path.basename(args.eval_path).replace(".jsonl", "")

pipe_tag = args.pipeline if args.pipeline == "standard" else f"{args.pipeline}_n{args.iter_num}"

filename_prefix_base = (
  f"{model_name}_"
  f"{dataset_name}_"
  f"rag{int(args.use_rag)}_k{args.rag_top_k}_"
  f"comp{int(args.compress)}_{args.compress_method}_r{args.compression_rate}_"
  f"rerank{int(args.rerank)}_n{args.rerank_top_n}_"
  f"ret{ret_name}_{corpus_name}_{idx_name}_"
  f"batch{int(args.batch)}_{args.batch_size}_"
  f"pipe{pipe_tag}_"
  f"{dev_tag}"
)


timing_jsonl = f"{TIMING_DIR}/perfiii_{filename_prefix_base}.jsonl"
timing_csv   = f"{TIMING_DIR}/perfiii_{filename_prefix_base}.csv"
out_csv      = f"{TESTING_DIR}/results_{filename_prefix_base}.csv"


if os.path.exists(out_csv):
    print(f"[WARNING] Results already exist: {out_csv}")
    print("Exiting before loading models.")
    exit(0)


# -----------------------------------------------------------------------------
# Initialize other libraries
# -----------------------------------------------------------------------------
from tqdm import tqdm
import pandas as pd
from rag_system.embedding import EmbeddingModel
from rag_system.retrieval import Retriever
from rag_system.generation import Generator
from rag_system.compression import Compressor
from rag_system.scoring import rouge_l, f1_score, exact_match, retrieval_recall
from rag_system.utils import extract_answer
from rag_system.timing import (
    RAGTiming, profile_block,
    reset_gpu_peak, get_gpu_memory
)
from rag_system.pipelines import PIPELINES


# -----------------------------------------------------------------------------
# Library build diagnostics
# -----------------------------------------------------------------------------
import subprocess as _sp

print("\n===== Library Build Diagnostics =====")
print(f"PyTorch version:   {torch.__version__}")
print(f"CUDA version:      {torch.version.cuda}")
print(f"cuDNN version:     {torch.backends.cudnn.version()}")
print(f"cuDNN enabled:     {torch.backends.cudnn.enabled}")
print(f"MKL available:     {torch.backends.mkl.is_available()}")
print(f"OpenMP threads:    {torch.get_num_threads()}")
print(f"GPU count:         {torch.cuda.device_count()}")
for _i in range(torch.cuda.device_count()):
    _cap = torch.cuda.get_device_capability(_i)
    print(f"  GPU {_i}: {torch.cuda.get_device_name(_i)}  (compute {_cap[0]}.{_cap[1]})")

try:
    import faiss
    print(f"FAISS version:     {faiss.__version__}")
    print(f"FAISS GPU build:   {hasattr(faiss, 'StandardGpuResources')}")
except Exception as _e:
    print(f"FAISS:             import failed ({_e})")

try:
    import transformers
    print(f"Transformers:      {transformers.__version__}")
except Exception as _e:
    print(f"Transformers:      import failed ({_e})")

try:
    import bitsandbytes as _bnb
    print(f"bitsandbytes:      {_bnb.__version__}")
    _bnb_cuda = getattr(_bnb, "COMPILED_WITH_CUDA", None)
    print(f"  compiled w/ CUDA:{_bnb_cuda}")
except Exception as _e:
    print(f"bitsandbytes:      not available ({_e})")

try:
    _pt_conf = torch.__config__.show()
    _relevant = [l for l in _pt_conf.splitlines()
                 if any(k in l for k in ("AVX", "AVX2", "AVX512", "NCCL", "OpenMP", "MKL", "CUDA", "cuDNN", "BUILD_TYPE"))]
    print("\nPyTorch build flags:")
    for _l in _relevant:
        print(" ", _l.strip())
except Exception as _e:
    print(f"Could not read PyTorch build config: {_e}")

try:
    _r = _sp.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                  "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    if _r.returncode == 0:
        print("\nnvidia-smi GPU summary:")
        for _l in _r.stdout.strip().splitlines():
            print(" ", _l)
except Exception as _e:
    print(f"nvidia-smi query failed: {_e}")

print("=====================================\n")


# -----------------------------------------------------------------------------
# Initial timing capture
# -----------------------------------------------------------------------------
init_timing = RAGTiming()

gpu_info = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "num_gpus": torch.cuda.device_count(),
}

for i in range(torch.cuda.device_count()):
    gpu_info[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)

init_timing.data.update(gpu_info)

# Pipeline metadata (for standardized comparison across runs)
init_timing.data["pipeline"] = args.pipeline
init_timing.data["iter_num"] = args.iter_num
init_timing.data["rerank"] = args.rerank
init_timing.data["rerank_model"] = args.rerank_model
init_timing.data["rerank_top_n"] = args.rerank_top_n

# Initial memory snapshot (all GPUs)
init_timing.capture_cpu_memory("init_cpu_start")
init_timing.capture_all_gpu_memory("init_gpu_start")


# -----------------------------------------------------------------------------
# Load embedding model - why is this only capturing cpu memory?
# -----------------------------------------------------------------------------
print("\nLoading embedding model...")

init_timing.capture_cpu_memory("load_embedder_cpu_before")

embedder = EmbeddingModel(args.retrieval_model, )
#embed_device = getattr(embedder, "device", None)
embed_device = EMBED_DEVICE

init_timing.capture_cpu_memory("load_embedder_cpu_after")

if embed_device is not None:
    init_timing.reset_gpu_peak(embed_device)
    init_timing.capture_gpu_memory("load_embedder_gpu_after", embed_device)

init_timing.data["embedder_device"] = str(embed_device)

if args.use_rag:
    # -----------------------------------------------------------------------------
    # Load FAISS corpus (CPU)
    # -----------------------------------------------------------------------------
    print("Loading FAISS corpus...")

    init_timing.capture_cpu_memory("load_corpus_cpu_before")

    with open(args.retrieval_json, "r") as f:
        corpus = [json.loads(line)["text"] for line in f]

    init_timing.capture_cpu_memory("load_corpus_cpu_after")

    # FAISS corpus is CPU-resident, but record GPU snapshot for completeness
    init_timing.capture_all_gpu_memory("load_corpus_gpu_after")


    # -----------------------------------------------------------------------------
    # Load FAISS index
    # -----------------------------------------------------------------------------


if args.use_rag:

    print("Loading FAISS index...")
    init_timing.capture_all_gpu_memory("load_faiss_gpu_before")
    init_timing.capture_all_nvml_memory("load_faiss_nvml_before")
    init_timing.capture_cpu_memory("load_faiss_cpu_before")

    print("embed_device:", embed_device)
    print("embed_device.index:", embed_device.index)
    print("type(embed_device.index):", type(embed_device.index))

    retriever = Retriever(
        args.retrieval_index,
        corpus,
        gpu_ids=INDEX_GPU_IDS
    )

    init_timing.capture_cpu_memory("load_faiss_cpu_after")

    # If FAISS uses GPU internally, this will show it
    init_timing.capture_all_gpu_memory("load_faiss_gpu_after")
    init_timing.capture_all_nvml_memory("load_faiss_nvml_after")

    print(f"[Retriever] Corpus size: {len(corpus)}, FAISS ntotal: {retriever.index.ntotal}")

    init_timing.data["faiss_is_gpu"] = retriever.is_gpu_index
    init_timing.data["faiss_device"] = str(retriever.device)


#########################temp code###########################################


import subprocess


if args.use_rag:

    print("\nFAISS Diagnostics --------------------")

    # 1. Confirm index type
    try:
        print("FAISS index object:", retriever.index)
        print("FAISS ntotal:", retriever.index.ntotal)
    except Exception as e:
        print("Could not access retriever.index:", e)

    # 2. Torch GPU memory (what PyTorch sees)
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"[Torch] GPU {i} allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")

    # 3. What NVIDIA sees (true total GPU usage)
    print("\n[nvidia-smi output]")
    subprocess.run(["nvidia-smi"])
    print("------------------------------------------------\n")

# -----------------------------------------------------------------------------
# Load generator model
# -----------------------------------------------------------------------------
print("Loading generator model...")

init_timing.capture_cpu_memory("load_generator_cpu_before")

#generator = Generator(args.model_path)
generator = Generator(args.model_path, device=GEN_DEVICE)
gen_device = getattr(generator, "device", None)

init_timing.capture_cpu_memory("load_generator_cpu_after")

if gen_device is not None:
    init_timing.reset_gpu_peak(gen_device)
    init_timing.capture_gpu_memory("load_generator_gpu_after", gen_device)

init_timing.data["generator_device"] = str(gen_device)


RERANK_DEVICE = parse_device(args.rerank_device)

reranker = None
if args.rerank:
    from rag_system.reranking import Reranker
    print(f"Reranking enabled: {args.rerank_model} top_n={args.rerank_top_n}")
    init_timing.capture_gpu_memory("load_reranker_gpu_before", RERANK_DEVICE)
    reranker = Reranker(args.rerank_model, device=RERANK_DEVICE)
    init_timing.capture_gpu_memory("load_reranker_gpu_after", RERANK_DEVICE)

compressor = None
if args.compress:
    print(f"Compression enabled: {args.compress_method} with rate {args.compression_rate}")

    init_timing.capture_cpu_memory("load_compressor_cpu_before")
    init_timing.capture_gpu_memory("load_compressor_gpu_before", COMPRESS_DEVICE)

    compressor = Compressor(args.compress_method, args.compression_rate)

    init_timing.capture_cpu_memory("load_compressor_cpu_after")
    init_timing.capture_gpu_memory("load_compressor_gpu_after", COMPRESS_DEVICE)

print("\nLoading evaluation dataset...")
with open(args.eval_path, "r") as f:
    eval_data = [json.loads(line) for line in f]


# Set seed for reproducibility
random.seed(55)

# Sample without replacement
eval_data = random.sample(eval_data, args.eval_size)


# -----------------------------------------------------------------------------
# Device registry
# -----------------------------------------------------------------------------
device_info = {}

# Embedder
embed_device = getattr(embedder, "device", None)
device_info["embedder_device"] = str(embed_device)

# Generator
gen_device = getattr(generator, "device", None)
device_info["generator_device"] = str(gen_device)

# Compressor (may be CPU-only)
compress_device = getattr(compressor, "device", None) if compressor else None
device_info["compressor_device"] = str(compress_device)

# FAISS
if args.use_rag:
    faiss_device = getattr(retriever, "device", None)
    device_info["faiss_device"] = str(faiss_device)
    device_info["faiss_is_gpu"] = retriever.is_gpu_index

# Persist once in init timing
init_timing.data.update(device_info)

print(f"\nBuilding pipeline: {args.pipeline}...")
pipeline = PIPELINES[args.pipeline](
    embedder=embedder,
    retriever=retriever if args.use_rag else None,
    generator=generator,
    compressor=compressor,
    reranker=reranker,
    args=args,
    embed_device=embed_device,
    gen_device=gen_device,
    index_device=INDEX_DEVICE,
)

# -----------------------------------------------------------------------------
# MAIN EVALUATION LOOP
# -----------------------------------------------------------------------------
results = []
timing_rows = []



if not args.batch:

    for idx, item in tqdm(enumerate(eval_data), total=len(eval_data)):
        q = item["question"]
        gold = item["golden_answers"]

        filename_prefix = f"{filename_prefix_base}_single_{idx}"

        pred, docs, raw, timing = pipeline.run_single(q)

        # compute scores
        r = rouge_l(pred, gold)
        f = f1_score(pred, gold)
        e = exact_match(pred, gold)
        rc = retrieval_recall(docs, gold) if args.use_rag else None

        results.append([gold, pred, r, f, e, rc])

        # save timing
        timing.data["question"] = q
        timing.data["prediction"] = pred
        timing.data["mode"] = "single"
        timing.data["idx"] = idx
        timing.data.update(init_timing.data)
        timing_rows.append(timing.data)



else:
    B = args.batch_size

    for start in tqdm(range(0, len(eval_data), B)):
        batch = eval_data[start:start+B]
        questions = [item["question"] for item in batch]
        golds = [item["golden_answers"] for item in batch]

        filename_prefix = f"{filename_prefix_base}_batch_{start}"

        preds, docs_batch, timing = pipeline.run_batch(questions)

        for pred, gold, docs in zip(preds, golds, docs_batch if args.use_rag else [[]]*len(preds)):
            r = rouge_l(pred, gold)
            f = f1_score(pred, gold)
            e = exact_match(pred, gold)
            rc = retrieval_recall(docs, gold) if args.use_rag else None
            results.append([gold, pred, r, f, e, rc])


        timing.data["mode"] = "batch"
        timing.data["batch_start"] = start
        timing.data["batch_size"] = len(preds)
        timing.data.update(init_timing.data)
        timing_rows.append(timing.data)



# ------------------------------------------------------------------
# SAVE TIMING (ONCE PER EXPERIMENT)
# ------------------------------------------------------------------
timing_df = pd.DataFrame(timing_rows)

timing_df.to_json(timing_jsonl, orient="records", lines=True)
timing_df.to_csv(timing_csv, index=False)


# -----------------------------------------------------------------------------
# SAVE RESULTS CSV
# -----------------------------------------------------------------------------
df = pd.DataFrame(results, columns=["gold", "prediction", "rouge", "f1", "em", "recall"])

out_csv = f"{TESTING_DIR}/results_{filename_prefix_base}.csv"

df.to_csv(out_csv, index=False)

print(f"\nDONE! Saved results to {out_csv}")
print(f"Timing traces and JSONL saved to timingevals/")
