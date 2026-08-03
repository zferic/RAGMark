import subprocess
import os
import sys

# -----------------------------
# CONFIG: where main.py writes outputs
# -----------------------------
from config import BASE_OUT, base_index_data, data_index2018, DATASET_DIR
TESTING_DIR = os.path.join(BASE_OUT, "testingevals")
TIMING_DIR  = os.path.join(BASE_OUT, "timingevals")

#os.makedirs(TESTING_DIR, exist_ok=True)
#os.makedirs(TIMING_DIR, exist_ok=True)


def build_filename_prefix(
    model_path: str,
    eval_path: str,
    use_rag: bool,
    compress: bool,
    compress_rate: float,
    top_k: int,
    retrieval_model: str,
    retrieval_index: str,
    retrieval_json: str,
    compress_method: str,
    eval_size: int,
    batch: bool,
    batch_size: int,
    pipeline: str = "standard",
    iter_num: int = 3,
    rerank: bool = False,
    rerank_top_n: int = 3,
) -> str:
    """
    Build a unique, stable filename prefix that includes retrieval identity too,
    so different retrievers don't overwrite each other.
    """
    safe_model = model_path.replace("/", "_")
    dataset_name = os.path.basename(eval_path).replace(".jsonl", "")

    ret_name = retrieval_model.replace("/", "_")
    idx_name = os.path.basename(retrieval_index).replace(".", "_")
    corpus_name = os.path.basename(retrieval_json).replace(".jsonl", "")

    pipe_tag = f"{pipeline}" if pipeline == "standard" else f"{pipeline}_n{iter_num}"

    return (
        f"{safe_model}_"
        f"{dataset_name}_"
        f"eval{eval_size}_"
        f"rag{int(use_rag)}_k{top_k}_"
        f"comp{int(compress)}_{compress_method}_r{compress_rate}_"
        f"rerank{int(rerank)}_n{rerank_top_n}_"
        f"batch{int(batch)}_bs{batch_size}_"
        f"pipe{pipe_tag}_"
        f"ret{ret_name}_{corpus_name}_{idx_name}"
    )


def run_testing_script(
    model_path: str,
    eval_path: str,
    use_rag: bool,
    compress: bool,
    compress_rate: float,
    top_k: int,
    ret_model: str,
    index_name: str,
    jsonl_file_path: str,
    compress_method: str,
    eval_size: int = 400,
    batch: bool = True,
    batch_size: int = 8,
    pipeline: str = "standard",
    iter_num: int = 3,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_n: int = 3,
):
    """
    Run main.py with real booleans via store_true flags and stream output live.
    """
    command = [
        "python", "-u", "main.py",
        "--model_path", model_path,
        "--eval_path", eval_path,
        "--eval_size", str(eval_size),
        "--rag_top_k", str(top_k),
        "--compression_rate", str(compress_rate),
        "--batch_size", str(batch_size),
        "--retrieval_model", ret_model,
        "--retrieval_index", index_name,
        "--retrieval_json", jsonl_file_path,
        "--compress_method", compress_method,
        "--pipeline", pipeline,
        "--iter_num", str(iter_num),
    ]

    # store_true flags: add the flag only when True
    if use_rag:
        command.append("--use_rag")
    if compress:
        command.append("--compress")
    if batch:
        command.append("--batch")
    if rerank:
        command.extend(["--rerank", "--rerank_model", rerank_model, "--rerank_top_n", str(rerank_top_n)])

    print("Command:", " ".join(command))


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
        print(f"Error running model {model_path} with dataset {eval_path}: {e}")
        return 1


# -----------------------------
# Define models / data
# -----------------------------
gen_models = [
   # "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
  # "meta-llama/Llama-3.2-1B-Instruct",
  # "meta-llama/Llama-3.2-3B-Instruct",
 #   "Qwen/Qwen2.5-7B-Instruct",
   # "Qwen/Qwen2.5-3B-Instruct",
]

# -----------------------------
# Define Data Index and Embedding pairs
# -----------------------------

ret_models = [
    ["intfloat/e5-base-v2",  os.path.join(base_index_data, "e5-base-v2-2018", "index", "e5_Flat.index"), data_index2018],
    ["intfloat/e5-small-v2", os.path.join(base_index_data, "e5-small-v2-2018", "index", "e5_Flat.index"), data_index2018],
    ["intfloat/e5-large-v2", os.path.join(base_index_data, "index2018-e5-large", "index", "e5_Flat.index"), data_index2018]
]


"""
ret_models = [
    ["intfloat/e5-large-v2", os.path.join(base_index_data, "index2018-e5-large", "index", "e5_Flat.index"), data_index2018]
   
]
"""


# -----------------------------
# Define Evaluation Data
# -----------------------------
##
datasets = [
    os.path.join(DATASET_DIR, "squad_dataset.jsonl"),
    os.path.join(DATASET_DIR, "nq_dataset.jsonl"),
    os.path.join(DATASET_DIR, "triviaqa_dataset.jsonl"),
    os.path.join(DATASET_DIR, "popqa_dataset.jsonl"),
    os.path.join(DATASET_DIR, "webquestions_dataset.jsonl"),
]

# Now these are real booleans
userags = [True,False]
compresss = [True,False]          # set to [False] to disable compression
batches  = [True, False]           # set to [False] for single mode

compress_methods = ["longllmlingua", "llmlingua2", "sc"]

# only meaningful when compress=True
#compress_rates = [.2, .4, .6, .8]
compress_rates = [0.2]
top_ks = [10]
#top_ks = [5]

EVAL_SIZE = 400
BATCH_SIZE = 8

#pipelines = [("standard", 1), ("iterative", 3)]  # (pipeline_name, iter_num)
pipelines = [("standard", 1)]

reranks        = [True, False]
rerank_top_ns  = [3]
rerank_models  = ["cross-encoder/ms-marco-MiniLM-L-6-v2"]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"


# -----------------------------
# Run grid
# -----------------------------
for model in gen_models:
    for ret_model, index_name, jsonl_file_path in ret_models:
        for dataset in datasets:
            for use_rag in userags:

                for compress in (compresss if use_rag else [False]):
                    for batch in batches:

                        # choose rates/methods depending on compression
                        if not compress:
                            run_methods = ["none"]
                            run_rates = [1.0]
                        else:
                            run_methods = compress_methods
                            run_rates = compress_rates

                        for compress_method in run_methods:
                            for compress_rate in run_rates:

                                run_top_ks = top_ks if use_rag else [0]

                                for top_k in run_top_ks:
                                    for pipeline, iter_num in pipelines:
                                        for rerank in (reranks if use_rag else [False]):
                                            for rerank_top_n in (rerank_top_ns if rerank else [0]):
                                                for rerank_model in (rerank_models if rerank else [rerank_models[0]]):

                                                    filename_prefix = build_filename_prefix(
                                                        model_path=model,
                                                        eval_path=dataset,
                                                        use_rag=use_rag,
                                                        compress=compress,
                                                        compress_rate=compress_rate,
                                                        top_k=top_k,
                                                        retrieval_model=ret_model,
                                                        retrieval_index=index_name,
                                                        retrieval_json=jsonl_file_path,
                                                        compress_method=compress_method,
                                                        eval_size=EVAL_SIZE,
                                                        batch=batch,
                                                        batch_size=BATCH_SIZE,
                                                        pipeline=pipeline,
                                                        iter_num=iter_num,
                                                        rerank=rerank,
                                                        rerank_top_n=rerank_top_n,
                                                    )

                                                    results_csv = os.path.join(TESTING_DIR, f"results_{filename_prefix}.csv")
                                                    perf_jsonl  = os.path.join(TIMING_DIR,  f"perfiii_{filename_prefix}.jsonl")
                                                    perf_csv    = os.path.join(TIMING_DIR,  f"perfiii_{filename_prefix}.csv")

                                                    if all(os.path.exists(p) for p in [results_csv, perf_jsonl, perf_csv]):
                                                        print(f"Skipping {filename_prefix} - outputs exist.")
                                                        continue

                                                    print(
                                                        f"\nRunning: model={model} dataset={os.path.basename(dataset)} "
                                                        f"rag={use_rag} topk={top_k} compress={compress} "
                                                        f"method={compress_method} rate={compress_rate} "
                                                        f"rerank={rerank} rerank_top_n={rerank_top_n} "
                                                        f"rerank_model={rerank_model} "
                                                        f"ret={ret_model} batch={batch} pipeline={pipeline}\n"
                                                        + "-" * 80
                                                    )

                                                    rc = run_testing_script(
                                                        model_path=model,
                                                        eval_path=dataset,
                                                        use_rag=use_rag,
                                                        compress=compress,
                                                        compress_rate=compress_rate,
                                                        top_k=top_k,
                                                        ret_model=ret_model,
                                                        index_name=index_name,
                                                        jsonl_file_path=jsonl_file_path,
                                                        compress_method=compress_method,
                                                        eval_size=EVAL_SIZE,
                                                        batch=batch,
                                                        batch_size=BATCH_SIZE,
                                                        pipeline=pipeline,
                                                        iter_num=iter_num,
                                                        rerank=rerank,
                                                        rerank_model=rerank_model,
                                                        rerank_top_n=rerank_top_n,
                                                    )

                                                    if rc != 0:
                                                        print(f"FAILED: Run failed (return code {rc}) for {filename_prefix}")
