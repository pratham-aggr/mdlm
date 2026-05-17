import re
import json
import random
import os
from datasets import load_dataset

N_PROBLEMS = 200
SEED       = 42
OUT_PATH   = os.path.join(os.path.dirname(__file__), "gsm8k_200.jsonl")


def extract_answer(solution: str) -> int:
    # GSM8K answers follow the pattern "#### <integer>"
    match = re.search(r"####\s*([\d,]+)", solution)
    if match is None:
        raise ValueError(f"No #### answer found in: {solution!r}")
    return int(match.group(1).replace(",", ""))


dataset = load_dataset("gsm8k", "main", split="test")
print(f"Loaded GSM8K test split: {len(dataset)} problems")

rng = random.Random(SEED)
indices = rng.sample(range(len(dataset)), N_PROBLEMS)
indices.sort()

records = []
for i, idx in enumerate(indices):
    row = dataset[idx]
    records.append({
        "id":       i,
        "question": row["question"],
        "answer":   extract_answer(row["answer"]),
    })

with open(OUT_PATH, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"Saved {N_PROBLEMS} problems → {OUT_PATH}")
print()
print("First 3 problems:")
for rec in records[:3]:
    print(f"  [{rec['id']}] Q: {rec['question'][:80]}...")
    print(f"        A: {rec['answer']}")
    print()
