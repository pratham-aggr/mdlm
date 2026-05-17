import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import json
import torch
from model import load_model
from generate import generate_with_measurements
from conditions import CONDITIONS
import config

with open(os.path.join(config.DATA_DIR, "harmful_prompts.json")) as f:
    HARMFUL_TEST = json.load(f)[:3]

with open(os.path.join(config.DATA_DIR, "harmless_prompts.json")) as f:
    HARMLESS_TEST = json.load(f)[:3]

model, tokenizer, MASK_ID = load_model()
v = torch.load(os.path.join(config.RESULTS_DIR, "steering_vector.pt")).to(next(model.parameters()).device)

CONDITIONS_TO_TEST = [
    ('A', CONDITIONS['A'], None),   # baseline — no steering
    ('B', CONDITIONS['B'], v),      # full steering — all positions, all steps
    ('I', CONDITIONS['I'], v),      # proposed fix — prompt only, last 20% steps
]

for label, prompts in [("HARMFUL", HARMFUL_TEST), ("HARMLESS", HARMLESS_TEST)]:
    print(f"\n{'='*60}")
    print(f"{label} PROMPTS")
    print('='*60)
    for prompt in prompts:
        print(f"\nPrompt: {prompt[:60]}")
        for cond_key, cond, sv in CONDITIONS_TO_TEST:
            _, tokens = generate_with_measurements(
                model, tokenizer, MASK_ID,
                prompt, cond, sv, config
            )
            response = "".join(tokens)[:150].replace('\n', ' ')
            print(f"  [{cond_key}] {response}")
