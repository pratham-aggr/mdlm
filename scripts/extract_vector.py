import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import json
import torch
from model import load_model
from steering import _get_layer
import config

LAYER_IDX      = 23
TOKEN_POSITION = -1   # last token of the formatted prompt


def collect_activations(model, tokenizer, prompts, layer_idx, token_position):
    """
    Run a forward pass on each prompt and capture the hidden state at
    token_position from the specified layer.

    Returns tensor of shape (n_prompts, d_model) in float32.
    """
    device = next(model.parameters()).device
    acts   = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt",
                           truncation=True, max_length=512).to(device)

        captured = {}

        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["hs"] = hs.detach()
            return None  # read-only

        handle = _get_layer(model, layer_idx).register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        hs = captured["hs"]                        # (1, seq_len, d_model)
        acts.append(hs[0, token_position, :].float())  # (d_model,)

    return torch.stack(acts)                       # (n_prompts, d_model)


# ------------------------------------------------------------------ load model
model, tokenizer, MASK_ID = load_model()

# ----------------------------------------------------------- load prompt lists
with open("data/harmful_prompts.json")  as f: harmful_prompts  = json.load(f)
with open("data/harmless_prompts.json") as f: harmless_prompts = json.load(f)

print(f"Harmful prompts:  {len(harmful_prompts)}")
print(f"Harmless prompts: {len(harmless_prompts)}")

# --------------------------------------------------- collect activations
print(f"\nCollecting activations at layer {LAYER_IDX}, position {TOKEN_POSITION} ...")
H_plus  = collect_activations(model, tokenizer, harmful_prompts,  LAYER_IDX, TOKEN_POSITION)
H_minus = collect_activations(model, tokenizer, harmless_prompts, LAYER_IDX, TOKEN_POSITION)
print(f"H_plus  shape: {H_plus.shape}")
print(f"H_minus shape: {H_minus.shape}")

# ----------------------------------------------------------- compute vector
mu_plus  = H_plus.mean(dim=0)
mu_minus = H_minus.mean(dim=0)
diff     = mu_plus - mu_minus
v        = diff / diff.norm()

print(f"\nmu_plus  norm: {mu_plus.norm():.2f}")
print(f"mu_minus norm: {mu_minus.norm():.2f}")
print(f"diff     norm: {diff.norm():.2f}")
print(f"v        norm: {v.norm():.6f}  (should be 1.0)")

# -------------------------------------------------------------- sanity check
print("\nCosine similarity with 5 random unit vectors (expect ~0):")
for i in range(5):
    rand = torch.randn_like(v)
    rand = rand / rand.norm()
    sim  = (v * rand).sum().item()
    print(f"  cos_sim(v, random_{i}): {sim:.4f}  (expect ~0)")

# --------------------------------------------------------------- save vector
os.makedirs(config.RESULTS_DIR, exist_ok=True)
out_path = os.path.join(config.RESULTS_DIR, "steering_vector.pt")
torch.save(v, out_path)
print(f"\nSaved steering vector → {out_path}")
