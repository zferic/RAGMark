import json
import numpy as np
from transformers import AutoTokenizer
name = "/projects/nucar/feric.z/FLashRAG-z/test_sample_2018_sent.jsonl"

with open(name , "r") as f:
    eval_data = [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained("gpt2")

token_counts = [len(tokenizer.encode(row["text"])) for row in eval_data]
token_counts = np.array(token_counts)

print(f"n       : {len(token_counts)}")
print(f"mean    : {token_counts.mean():.1f}")
print(f"std     : {token_counts.std():.1f}")
print(f"min     : {token_counts.min()}")
print(f"max     : {token_counts.max()}")
print(f"median  : {np.median(token_counts)}")


