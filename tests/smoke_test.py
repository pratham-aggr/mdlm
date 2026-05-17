import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import json
import config
from model import load_model
from generate import generate_with_measurements

# --- load model
model, tokenizer, MASK_ID = load_model()

# --- load first GSM8K problem
with open("data/gsm8k_200.jsonl") as f:
    problem = json.loads(f.readline())
print(f"Problem [{problem['id']}]: {problem['question'][:80]}...")
print(f"Expected answer: {problem['answer']}\n")

# --- condition A: baseline, no steering
CONDITION_A = {
    'steer_layers':       [],
    'steer_positions':    'all',
    'steer_when':         'all',
    'save_hidden_states': False,
}

# --- run
measurements, response_tokens = generate_with_measurements(
    model, tokenizer, MASK_ID,
    prompt_str=problem['question'],
    condition=CONDITION_A,
    steering_vector=None,
    config=config,
)

# --- diagnostics
# 1. entropy should decrease over steps
print("entropy_per_step first 5:", measurements['entropy_per_step'][:5])
print("entropy_per_step last 5: ", measurements['entropy_per_step'][-5:])
assert measurements['entropy_per_step'][0] > measurements['entropy_per_step'][-1], \
    "Entropy should decrease over denoising steps"

# 2. all positions should be unmasked by end
assert (measurements['unmask_step'] != -1).all(), \
    f"{(measurements['unmask_step'] == -1).sum()} positions never unmasked"

# 3. confidence should be reasonable
mean_conf = measurements['confidence_at_unmask'].mean()
print(f"mean confidence at unmask: {mean_conf:.3f}")
assert 0.3 < mean_conf < 1.0, f"Suspicious mean confidence: {mean_conf}"

# 4. entropy_at_commit should be non-zero
mean_entropy = measurements['entropy_at_commit'].mean()
print(f"mean entropy at commit: {mean_entropy:.3f}")
assert mean_entropy > 0.0, "entropy_at_commit is all zeros"

# 5. print decoded response
print("\n--- Response ---")
print("".join(response_tokens))
print("--- End ---")

print("\nAll assertions passed.")
