import os

# Edit these paths when moving to a new system or cluster

HF_HOME = "/projects/nucar/feric.z/hf_home"

BASE_OUT   = "/scratch/feric.z/ragbench_outputs"
OUT_SUFFIX = "_a100"

WIKI_INDEX_DIR   = "/projects/nucar/feric.z/wikiindex2/"
WIKI_CORPUS_2018 = "/projects/nucar/feric.z/FLashRAG-z/test_sample_2018_sent.jsonl"

# Short aliases matching variable names used throughout the codebase
base_index_data = WIKI_INDEX_DIR
data_index2018  = WIKI_CORPUS_2018

DATASET_DIR = "/home/feric.z/FlashRAG/examples/quick_start/dataset"
