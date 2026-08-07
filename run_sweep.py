"""
run_sweep.py

Loads embedder, FAISS index, generator, and eval data ONCE,
then iterates over a list of (compress_method, rate, topk, use_rag) configs.
Called by run_sweep_fig4_fig9.py or run_sweep_fig5.py with --sweep_configs as a JSON array.
"""

import argparse
import json
import os
import random
import re
import hashlib
import torch

# -------------------------------------------------------------------------
# HF cache setup
# -------------------------------------------------------------------------
from config import HF_HOME, BASE_OUT
os.makedirs(HF_HOME, exist_ok=True)
for k in ["TRANSFORMERS_CACHE","HF_HOME","HF_HUB_CACHE","HUGGINGFACE_HUB_CACHE",
          "XDG_CACHE_HOME","HF_ASSETS_CACHE","HF_DATASETS_CACHE","HF_MODULES_CACHE"]:
    os.environ.pop(k, None)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"]           = os.path.join(HF_HOME, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"]  = os.path.join(HF_HOME, "hub")
os.environ["HF_ASSETS_CACHE"]        = os.path.join(HF_HOME, "assets")
os.environ["HF_DATASETS_CACHE"]      = os.path.join(HF_HOME, "datasets")
os.environ["HF_MODULES_CACHE"]       = os.path.join(HF_HOME, "modules")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token

# -------------------------------------------------------------------------
# Args
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Fixed (loaded once)
parser.add_argument("--model_path",       type=str, required=True)
parser.add_argument("--retrieval_model",  type=str, required=True)
parser.add_argument("--retrieval_index",  type=str, required=True)
parser.add_argument("--retrieval_json",   type=str, required=True)
parser.add_argument("--eval_paths",       type=str, required=True,
                    help='JSON array of dataset paths')
parser.add_argument("--eval_size",        type=int, default=400)
parser.add_argument("--batch",            action="store_true")
parser.add_argument("--batch_size",       type=int, default=8)
parser.add_argument("--embed_device",     type=str, default="cuda:0")
parser.add_argument("--index_device",     type=str, default="cuda:0,cuda:1")
parser.add_argument("--compress_device",  type=str, default="cuda:0")
parser.add_argument("--gen_device",       type=str, default="cuda:0")
parser.add_argument("--rerank",           action="store_true")
parser.add_argument("--rerank_model",     type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
parser.add_argument("--rerank_top_n",     type=int, default=3)
parser.add_argument("--rerank_device",    type=str, default="cuda:0")

# Sweep configs: JSON array of dicts with keys:
#   use_rag, compress, compress_method, compression_rate, rag_top_k
parser.add_argument("--sweep_configs", type=str, required=True,
                    help='JSON array of config dicts')
# Output
parser.add_argument("--base_out", type=str,
                    default=BASE_OUT)
parser.add_argument("--out_suffix", type=str, default="",
                    help='e.g. "_a100" appended to testingevals/timingevals dir names')
parser.add_argument("--pipeline", type=str, default="standard",
                    choices=["standard", "iterative"])

args = parser.parse_args()

sweep_configs = json.loads(args.sweep_configs)
eval_paths    = json.loads(args.eval_paths)

TESTING_DIR = f"{args.base_out}/testingevals{args.out_suffix}"
TIMING_DIR  = f"{args.base_out}/timingevals{args.out_suffix}"

os.makedirs(TESTING_DIR, exist_ok=True)
os.makedirs(TIMING_DIR,  exist_ok=True)

# -------------------------------------------------------------------------
# Device parsing
# -------------------------------------------------------------------------
def parse_device(s):
    s = str(s)
    if s.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(s)

EMBED_DEVICE    = parse_device(args.embed_device)
COMPRESS_DEVICE = parse_device(args.compress_device)
RERANK_DEVICE   = parse_device(args.rerank_device)

_index_parts = [s.strip() for s in args.index_device.split(",")]
if _index_parts == ["cpu"]:
    INDEX_DEVICE  = torch.device("cpu")
    INDEX_GPU_IDS = [-1]
else:
    INDEX_GPU_IDS = [torch.device(p).index or 0 for p in _index_parts]
    INDEX_DEVICE  = torch.device(_index_parts[0])

# --gen_device accepts a comma-separated GPU list (e.g. "cuda:0,cuda:1") to shard the generator
# across multiple GPUs via device_map="auto", same convention as --index_device for FAISS sharding.
# GEN_DEVICE stays the single primary device used for tensor placement / memory sampling elsewhere;
# GEN_GPU_IDS is the full list, only consumed by Generator() when sharding.
_gen_parts = [s.strip() for s in args.gen_device.split(",")]
if _gen_parts == ["cpu"]:
    GEN_DEVICE  = torch.device("cpu")
    GEN_GPU_IDS = [-1]
else:
    GEN_GPU_IDS = [torch.device(p).index or 0 for p in _gen_parts]
    GEN_DEVICE  = torch.device(_gen_parts[0])

# -------------------------------------------------------------------------
# Filename helpers
# -------------------------------------------------------------------------
def _safe(s, max_len=180):
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace(":", "-").replace("/", "-")
    s = re.sub(r"[^A-Za-z0-9._=-]", "-", s)
    return s[:max_len].strip("._-")

def device_tag():
    full = (f"dev_embed={EMBED_DEVICE}_index={INDEX_DEVICE}"
            f"_indexgpu={'-'.join(str(i) for i in INDEX_GPU_IDS)}"
            f"_compress={COMPRESS_DEVICE}_gen={GEN_DEVICE}"
            f"_gengpu={'-'.join(str(i) for i in GEN_GPU_IDS)}"
            f"_rerank={RERANK_DEVICE}")
    return "dev" + hashlib.md5(_safe(full).encode()).hexdigest()[:6]

dev_tag     = device_tag()
ret_name    = os.path.basename(args.retrieval_model)
idx_name    = os.path.basename(args.retrieval_index).replace(".", "_")
corpus_name = os.path.basename(args.retrieval_json).replace(".jsonl", "")
model_name  = os.path.basename(args.model_path)

rerank_model_name = os.path.basename(args.rerank_model).replace("/", "-")

SEP = "~"

def make_prefix(cfg, dataset_name):
    return SEP.join([
        model_name,
        dataset_name,
        f"rag{int(cfg['use_rag'])}",
        f"k{cfg['rag_top_k']}",
        f"comp{int(cfg['compress'])}",
        cfg['compress_method'],
        f"r{cfg['compression_rate']}",
        f"rerank{int(args.rerank)}",
        f"n{args.rerank_top_n if args.rerank else 0}",
        rerank_model_name if args.rerank else "none",
        f"ret{ret_name}",
        corpus_name,
        idx_name,
        f"batch{int(args.batch)}",
        str(args.batch_size),
        f"pipe{args.pipeline}",
        dev_tag,
    ])

def outputs_exist(prefix):
    return all(os.path.exists(p) for p in [
        f"{TESTING_DIR}/results_{prefix}.csv",
        f"{TIMING_DIR}/perfiii_{prefix}.jsonl",
        f"{TIMING_DIR}/perfiii_{prefix}.csv",
    ])

# -------------------------------------------------------------------------
# Check if ALL configs for ALL datasets already done — exit early if so
# -------------------------------------------------------------------------
all_done = all(
    outputs_exist(make_prefix(cfg, os.path.basename(ep).replace(".jsonl", "")))
    for ep in eval_paths
    for cfg in sweep_configs
)
if all_done:
    print("All configs already complete. Nothing to do.")
    exit(0)

# -------------------------------------------------------------------------
# Imports (deferred so skip-check above is fast)
# -------------------------------------------------------------------------
from tqdm import tqdm
import pandas as pd
from rag_system.embedding import EmbeddingModel
from rag_system.retrieval import Retriever
from rag_system.generation import Generator
from rag_system.compression import Compressor
from rag_system.scoring import rouge_l, f1_score, exact_match, retrieval_recall
from rag_system.utils import extract_answer
from rag_system.timing import RAGTiming, PowerPoller, TracePoller
from rag_system.pipelines import PIPELINES

# -------------------------------------------------------------------------
# Load fixed components ONCE
# -------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SWEEP: {model_name} | {ret_name} | {len(eval_paths)} datasets")
print(f"{'='*60}\n")

init_timing = RAGTiming()

gpu_info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "num_gpus": torch.cuda.device_count()}
for i in range(torch.cuda.device_count()):
    gpu_info[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)
init_timing.data.update(gpu_info)
init_timing.capture_cpu_memory("init_cpu_start")
init_timing.capture_all_gpu_memory("init_gpu_start")

# Embedder
print("Loading embedder...")
init_timing.capture_cpu_memory("load_embedder_cpu_before")
embedder = EmbeddingModel(args.retrieval_model)
init_timing.capture_cpu_memory("load_embedder_cpu_after")
init_timing.capture_gpu_memory("load_embedder_gpu_after", EMBED_DEVICE)
init_timing.data["embedder_device"] = str(EMBED_DEVICE)

# Corpus + FAISS (only needed if any pending config uses RAG)
need_rag = any(cfg["use_rag"] for cfg in sweep_configs)
retriever = None
corpus = None
if need_rag:
    print("Loading corpus...")
    init_timing.capture_cpu_memory("load_corpus_cpu_before")
    with open(args.retrieval_json, "r") as f:
        corpus = [json.loads(line)["text"] for line in f]
    init_timing.capture_cpu_memory("load_corpus_cpu_after")
    init_timing.capture_all_gpu_memory("load_corpus_gpu_after")

    print("Loading FAISS index...")
    init_timing.capture_all_gpu_memory("load_faiss_gpu_before")
    init_timing.capture_all_nvml_memory("load_faiss_nvml_before")
    init_timing.capture_cpu_memory("load_faiss_cpu_before")
    retriever = Retriever(args.retrieval_index, corpus, gpu_ids=None if INDEX_GPU_IDS == [-1] else INDEX_GPU_IDS)
    init_timing.capture_cpu_memory("load_faiss_cpu_after")
    init_timing.capture_all_gpu_memory("load_faiss_gpu_after")
    init_timing.capture_all_nvml_memory("load_faiss_nvml_after")
    init_timing.data["faiss_is_gpu"] = retriever.is_gpu_index
    init_timing.data["faiss_device"] = str(retriever.device)
    print(f"  Corpus: {len(corpus)}, FAISS ntotal: {retriever.index.ntotal}")

# Generator
print("Loading generator...")
init_timing.capture_cpu_memory("load_generator_cpu_before")
generator = Generator(args.model_path, device=GEN_DEVICE,
                       gpu_ids=None if GEN_GPU_IDS == [-1] else GEN_GPU_IDS)
init_timing.capture_cpu_memory("load_generator_cpu_after")
init_timing.capture_gpu_memory("load_generator_gpu_after", GEN_DEVICE)
init_timing.data["generator_device"] = str(GEN_DEVICE)

reranker = None
if args.rerank:
    from rag_system.reranking import Reranker
    print(f"Loading reranker: {args.rerank_model} top_n={args.rerank_top_n}")
    init_timing.capture_gpu_memory("load_reranker_gpu_before", RERANK_DEVICE)
    reranker = Reranker(args.rerank_model, device=RERANK_DEVICE)
    init_timing.capture_gpu_memory("load_reranker_gpu_after", RERANK_DEVICE)
init_timing.data["rerank"]       = args.rerank
init_timing.data["rerank_model"] = args.rerank_model
init_timing.data["rerank_top_n"] = args.rerank_top_n

print(f"\nFixed components loaded. Starting dataset loop...\n")

# -------------------------------------------------------------------------
# Dataset loop — sweep over all configs for each dataset
# -------------------------------------------------------------------------
current_compressor_key = None  # (method, rate) — avoid reloading if unchanged
compressor = None

for eval_path in eval_paths:
    dataset_name = os.path.basename(eval_path).replace(".jsonl", "")
    pending = [cfg for cfg in sweep_configs if not outputs_exist(make_prefix(cfg, dataset_name))]

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"  {len(pending)}/{len(sweep_configs)} configs pending")
    print(f"{'='*60}")

    if not pending:
        print("  Already complete, skipping.")
        continue

    print("Loading eval data...")
    with open(eval_path, "r") as f:
        all_data = [json.loads(line) for line in f]
    random.seed(55)
    eval_data = random.sample(all_data, args.eval_size)

    for cfg_idx, cfg in enumerate(pending):
        prefix = make_prefix(cfg, dataset_name)

        use_rag         = cfg["use_rag"]
        compress        = cfg["compress"]
        compress_method = cfg["compress_method"]
        compress_rate   = float(cfg["compression_rate"])
        top_k           = int(cfg["rag_top_k"])

        print(f"\n[{cfg_idx+1}/{len(pending)}] "
              f"rag={use_rag} topk={top_k} compress={compress} "
              f"method={compress_method} rate={compress_rate}")
        print("-" * 60)

        # -- Swap compressor if needed --
        new_key = (compress_method, compress_rate) if compress else None
        if new_key != current_compressor_key:
            if compressor is not None:
                del compressor
                torch.cuda.empty_cache()
                compressor = None
            if compress:
                print(f"  Loading compressor: {compress_method} r={compress_rate}")
                compressor = Compressor(compress_method, compress_rate)
            current_compressor_key = new_key

        # -- Build pipeline --
        class _Cfg:
            pass
        pipe_args = _Cfg()
        pipe_args.use_rag         = use_rag
        pipe_args.compress        = compress
        pipe_args.compress_method = compress_method
        pipe_args.compression_rate= compress_rate
        pipe_args.rag_top_k       = top_k
        pipe_args.batch           = args.batch
        pipe_args.batch_size      = args.batch_size
        pipe_args.pipeline        = args.pipeline
        pipe_args.iter_num        = 2
        pipe_args.rerank_top_n    = args.rerank_top_n

        pipeline = PIPELINES[args.pipeline](
            embedder=embedder,
            retriever=retriever if use_rag else None,
            generator=generator,
            compressor=compressor if compress else None,
            reranker=reranker,
            args=pipe_args,
            embed_device=EMBED_DEVICE,
            gen_device=GEN_DEVICE,
            index_device=INDEX_DEVICE,
        )

        # -- Eval loop --
        results     = []
        timing_rows = []
        B           = args.batch_size

        if not args.batch:
            for idx, item in tqdm(enumerate(eval_data), total=len(eval_data)):
                q    = item["question"]
                gold = item["golden_answers"]
                poller = PowerPoller(interval=0.05)
                poller.start()
                # -- trace first 30 queries (same seed → same queries across all topk configs) --
                do_trace = (idx < 30)
                if do_trace:
                    tracer = TracePoller(interval=0.005)
                    tracer.start()
                pred, docs, raw, timing = pipeline.run_single(q, tracer=tracer if do_trace else None)
                poller.stop()
                if do_trace:
                    print("adding nth query", idx)
                    tracer.stop()
                    tracer.save(
                        f"{TIMING_DIR}/trace_{prefix}.jsonl",
                        meta={"prefix": prefix, "question": q, "mode": "single", "idx": idx}
                    )
                r  = rouge_l(pred, gold)
                f  = f1_score(pred, gold)
                e  = exact_match(pred, gold)
                rc = retrieval_recall(docs, gold) if use_rag else None
                results.append([gold, pred, r, f, e, rc])
                timing.data.update({"question": q, "prediction": pred,
                                     "mode": "single", "idx": idx,
                                     "energy_j":        poller.energy_joules(),
                                     "power_mean_w":    poller.mean_power_w(),
                                     "util_mean":       poller.mean_utilization(),
                                     "util_peak":       poller.peak_utilization()})
                timing.data.update(init_timing.data)
                timing_rows.append(timing.data)
        else:
            for start in tqdm(range(0, len(eval_data), B)):
                batch_items = eval_data[start:start+B]
                questions   = [x["question"]      for x in batch_items]
                golds       = [x["golden_answers"] for x in batch_items]
                poller = PowerPoller(interval=0.05)
                poller.start()
                # -- trace every 10th batch --
                do_trace = (start // B % 10 == 0)
                if do_trace:
                    tracer = TracePoller(interval=0.005)
                    tracer.start()
                preds, docs_batch, timing = pipeline.run_batch(questions, tracer=tracer if do_trace else None)
                poller.stop()
                if do_trace:
                    tracer.stop()
                    tracer.save(
                        f"{TIMING_DIR}/trace_{prefix}.jsonl",
                        meta={"prefix": prefix, "mode": "batch", "batch_start": start}
                    )
                docs_iter = docs_batch if use_rag else [[]] * len(preds)
                for pred, gold, docs in zip(preds, golds, docs_iter):
                    r  = rouge_l(pred, gold)
                    f  = f1_score(pred, gold)
                    e  = exact_match(pred, gold)
                    rc = retrieval_recall(docs, gold) if use_rag else None
                    results.append([gold, pred, r, f, e, rc])
                timing.data.update({"mode": "batch", "batch_start": start,
                                     "batch_size": len(preds),
                                     "energy_j":        poller.energy_joules(),
                                     "power_mean_w":    poller.mean_power_w(),
                                     "util_mean":       poller.mean_utilization(),
                                     "util_peak":       poller.peak_utilization()})
                timing.data.update(init_timing.data)
                timing_rows.append(timing.data)

        # -- Save --
        pd.DataFrame(timing_rows).to_json(
            f"{TIMING_DIR}/perfiii_{prefix}.jsonl", orient="records", lines=True)
        pd.DataFrame(timing_rows).to_csv(
            f"{TIMING_DIR}/perfiii_{prefix}.csv", index=False)
        pd.DataFrame(results, columns=["gold","prediction","rouge","f1","em","recall"]).to_csv(
            f"{TESTING_DIR}/results_{prefix}.csv", index=False)

        print(f"  Saved: {prefix}")

print("\nSweep complete.")
