import os

# Edit these paths when moving to a new system or cluster

HF_HOME = os.path.expanduser("~/.cache/huggingface")   # where HF models are downloaded/cached

BASE_OUT   = "/path/to/output/dir"   # where results CSVs and trace files are written
OUT_SUFFIX = "_local"                # appended to output subdirs (e.g. testingevals_local)

WIKI_INDEX_DIR   = "/path/to/wikiindex/"                      # root dir containing FAISS index subdirs
WIKI_CORPUS_2018 = "/path/to/test_sample_2018_sent.jsonl"     # Wikipedia 2018 corpus JSONL

# Short aliases matching variable names used throughout the codebase
base_index_data = WIKI_INDEX_DIR
data_index2018  = WIKI_CORPUS_2018

DATASET_DIR = "/path/to/flashrag/dataset"   # dir containing *_dataset.jsonl eval files
