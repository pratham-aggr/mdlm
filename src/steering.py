import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import functools
from typing import List
from config import LAYER_PATH


def _get_layer(model, layer_idx: int):
    blocks = functools.reduce(getattr, LAYER_PATH.split("."), model)
    return blocks[layer_idx]


def extract_steering_vector(
    model,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    layer: int,
    position_idx: int,
) -> torch.Tensor:
    """
    Compute a steering vector as the normalized mean-activation difference
    between harmful and harmless prompts at a given layer and token position.

    position_idx: negative index into the prompt token sequence
                  (-1 = last prompt token, i.e. right before the response).
    Returns v of shape (d_model,) on CUDA.
    """
    device = next(model.parameters()).device

    def get_activations(prompts: List[str]) -> torch.Tensor:
        acts = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            captured = {}

            def hook_fn(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured["hs"] = hs.detach()

            handle = _get_layer(model, layer).register_forward_hook(hook_fn)
            with torch.no_grad():
                model(**inputs)
            handle.remove()

            hs = captured["hs"]          # (1, seq_len, d_model)
            acts.append(hs[0, position_idx, :])  # (d_model,)

        return torch.stack(acts)         # (n_prompts, d_model)

    harmful_acts  = get_activations(harmful_prompts)   # (n, d_model)
    harmless_acts = get_activations(harmless_prompts)  # (n, d_model)

    mu_plus  = harmful_acts.mean(dim=0)   # (d_model,)
    mu_minus = harmless_acts.mean(dim=0)  # (d_model,)

    diff = mu_plus - mu_minus
    v = diff / diff.norm(p=2)
    return v  # (d_model,), on CUDA


def verify_layer_path(model, layer_idx: int) -> None:
    """
    Confirm that model.model.transformer.blocks[layer_idx] is the correct hook target.
    Prints the first 30 named modules so you can inspect the actual structure.
    """
    device = next(model.parameters()).device

    print("First 30 named modules:")
    for name, _ in list(model.named_modules())[:30]:
        print(f"  {name}")
    print()

    fired = {"ok": False}

    def hook_fn(module, input, output):
        fired["ok"] = True

    try:
        handle = _get_layer(model, layer_idx).register_forward_hook(hook_fn)
    except (AttributeError, IndexError) as e:
        print(f"Hook did not fire — inspect model structure  ({e})")
        return

    dummy = torch.zeros(1, 4, dtype=torch.long, device=device)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass
    handle.remove()

    if fired["ok"]:
        print(f"Hook fired correctly at {LAYER_PATH}[{layer_idx}]")
    else:
        print("Hook did not fire — inspect model structure")


def apply_steering(hidden_state, v, debug=False, alpha=None):
    """
    Steer hidden_state away from direction v.

    alpha=None  — projection mode: remove the component along v entirely.
    alpha=float — additive mode: subtract alpha * v_unit at every position,
                  pushing the representation away from the harmful direction.

    Args:
        hidden_state: (batch, seq_len, d_model) float16 or float32
        v:            (d_model,) the steering direction, any norm
        debug:        if True, print diagnostics and assert residual < 1e-5
        alpha:        scaling factor for additive mode; None means projection mode

    Returns:
        steered hidden_state, same dtype as input
    """
    orig_dtype = hidden_state.dtype

    h = hidden_state.float()                    # (batch, seq_len, d_model)
    v_unit = v.float().squeeze()                # (d_model,)
    assert v_unit.dim() == 1, \
        f"v must be 1D after squeeze, got {v_unit.shape}"
    v_unit = v_unit / v_unit.norm()             # ensure unit norm

    if alpha is not None:
        # additive mode: push hidden state in direction -v (away from harmful)
        result = h - alpha * v_unit.unsqueeze(0).unsqueeze(0)
    else:
        # projection mode: remove the component along v
        proj       = torch.einsum('bsd,d->bs', h, v_unit)   # (batch, seq_len)
        correction = torch.einsum('bs,d->bsd', proj, v_unit) # (batch, seq_len, d_model)
        result     = h - correction

        if debug:
            residual = torch.einsum('bsd,d->bs', result, v_unit).abs().max().item()
            norm_in  = h.norm().item()
            norm_out = result.norm().item()
            proj_abs = proj.abs()
            print(f"v_unit shape:     {v_unit.shape}")
            print(f"proj shape:       {proj.shape}")
            print(f"correction shape: {correction.shape}")
            print(f"proj  mean/max:   {proj_abs.mean():.4f} / {proj_abs.max():.4f}")
            print(f"correction norm:  {correction.norm():.4f}")
            print(f"residual:         {residual:.2e}  (target < 1e-5)")
            print(f"norm in → out:    {norm_in:.4f} → {norm_out:.4f}  "
                  f"(reduction: {norm_in - norm_out:.6f})")
            assert residual < 1e-5, f"Projection incomplete: residual={residual}"

    return result.to(orig_dtype)
