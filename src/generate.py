import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from steering import apply_steering, _get_layer


def _format_prompt(tokenizer, prompt_str):
    messages = [{"role": "user", "content": prompt_str}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _make_steer_hook(v, prompt_len, steer_positions, alpha=None):
    def hook_fn(module, input, output):
        is_tuple  = isinstance(output, tuple)
        hs        = output[0] if is_tuple else output   # (batch, seq_len, d_model)

        if steer_positions == 'all':
            hs_steered = apply_steering(hs, v, alpha=alpha)
        elif steer_positions == 'prompt_only':
            prompt_part   = apply_steering(hs[:, :prompt_len, :], v, alpha=alpha)
            response_part = hs[:, prompt_len:, :]
            hs_steered    = torch.cat([prompt_part, response_part], dim=1)
        elif steer_positions == 'response_only':
            prompt_part   = hs[:, :prompt_len, :]
            response_part = apply_steering(hs[:, prompt_len:, :], v, alpha=alpha)
            hs_steered    = torch.cat([prompt_part, response_part], dim=1)
        else:
            raise ValueError(f"Unknown steer_positions: {steer_positions!r}")

        return (hs_steered,) + output[1:] if is_tuple else hs_steered
    return hook_fn


def _make_read_hook(store, layer_idx, step):
    def hook_fn(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        store[(layer_idx, step)] = hs[0].detach().cpu().numpy()
        return None  # read-only: returning None leaves output unchanged
    return hook_fn


def generate_with_measurements(
    model, tokenizer, MASK_ID,
    prompt_str, condition, steering_vector, config,
):
    """
    Run one LLaDA generation pass with optional activation steering and
    per-step measurement collection.

    Parameters
    ----------
    condition : dict with keys
        steer_layers      : list[int]   — layers to apply steering hooks on
        steer_positions   : str         — 'all' | 'prompt_only' | 'response_only'
        steer_when        : str         — 'all' | 'first_20pct' | 'last_20pct'
        save_hidden_states: bool
    steering_vector : torch.Tensor (d_model,) or None
    config          : module with GEN_LEN, STEPS, SAVE_STEPS, SAVE_LAYERS

    Returns
    -------
    measurements : dict
    response_tokens : list[str]
    """
    device  = next(model.parameters()).device
    GEN_LEN = config.GEN_LEN
    STEPS   = config.STEPS
    # ------------------------------------------------------------------ setup
    formatted  = _format_prompt(tokenizer, prompt_str)
    inputs     = tokenizer(formatted, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
    prompt_ids = inputs["input_ids"]               # (1, prompt_len)
    prompt_len = prompt_ids.shape[1]

    x = torch.cat([
        prompt_ids,
        torch.full((1, GEN_LEN), MASK_ID, dtype=torch.long, device=device),
    ], dim=1)                                      # (1, prompt_len + GEN_LEN)

    # ---------------------------------------------------------- storage init
    measurements = {
        'confidence_at_unmask': np.zeros(GEN_LEN),
        'unmask_step':          np.full(GEN_LEN, -1, dtype=int),
        'entropy_per_step':     np.zeros(STEPS),
        'entropy_at_commit':    np.zeros(GEN_LEN),
    }
    if condition['save_hidden_states']:
        measurements['hidden_states'] = {}          # keyed by (layer_idx, step)

    # -------------------------------------------------------- denoising loop
    for step in range(STEPS):
        n_target  = int(GEN_LEN * (step + 1) / STEPS)
        n_already = int((x[0, prompt_len:] != MASK_ID).sum())
        n_unmask  = max(0, n_target - n_already)

        # ---- determine whether steering is active this step
        steer_when = condition.get('steer_when', 'all')
        steering_active = (
            steer_when == 'all'
            or (steer_when == 'first_20pct' and step < STEPS * 0.2)
            or (steer_when == 'last_20pct'  and step >= STEPS * 0.8)
        )
        
        # All layers share one hook instance — ok because v/prompt_len/steer_positions
        # are layer-invariant. If per-layer steering vectors are needed later,
        # move _make_steer_hook() call inside the for loop.

        # ---- register steering hooks (one per layer)
        steer_handles = []
        if steering_active and steering_vector is not None:
            hook = _make_steer_hook(
                steering_vector,
                prompt_len,
                condition.get('steer_positions', 'all'),
                alpha=condition.get('steering_alpha', None),
            )
            for layer_idx in condition['steer_layers']:
                steer_handles.append(
                    _get_layer(model, layer_idx).register_forward_hook(hook)
                )

        # ---- forward pass (steering hooks active if registered)
        try:
            with torch.no_grad():
                logits = model(x).logits           # (1, prompt_len + GEN_LEN, vocab_size)
        finally:
            for h in steer_handles:
                h.remove()

        # ---- extract response-region quantities
        # cast to float32 before softmax: float16 underflows to exact 0 for
        # very negative logits, causing log(0)=-inf and 0*-inf=nan in entropy
        resp_logits    = logits[0, prompt_len:].float()                # (GEN_LEN, vocab_size)
        log_probs      = F.log_softmax(resp_logits, dim=-1)            # stable log-probs
        probs          = log_probs.exp()                               # probs from log_probs
        best_p, best_t = probs.max(dim=-1)                             # (GEN_LEN,) each
        # clamp log_probs at -100 so zero-prob tokens contribute 0 not nan
        entropies      = -(probs * log_probs.clamp(min=-100)).sum(dim=-1)  # (GEN_LEN,)

        still_masked = (x[0, prompt_len:] == MASK_ID)                 # (GEN_LEN,)

        # ---- global avg entropy over still-masked positions
        measurements['entropy_per_step'][step] = (
            entropies[still_masked].mean().item() if still_masked.any() else 0.0
        )

        # ---- select top-n_unmask positions by confidence; record measurements
        if n_unmask > 0 and still_masked.any():
            best_p_sel = best_p.clone()
            best_p_sel[~still_masked] = -1.0
            _, top_idx = best_p_sel.topk(min(n_unmask, int(still_masked.sum())))

            for idx in top_idx.tolist():
                measurements['confidence_at_unmask'][idx] = best_p[idx].item()
                measurements['unmask_step'][idx]          = step
                measurements['entropy_at_commit'][idx]    = entropies[idx].item()

            x[0, prompt_len + top_idx] = best_t[top_idx]

        # ---- optional hidden-state capture (separate, read-only forward pass)
        if condition['save_hidden_states'] and step in config.SAVE_STEPS:
            read_handles = []
            for layer_idx in config.SAVE_LAYERS:
                read_handles.append(
                    _get_layer(model, layer_idx).register_forward_hook(
                        _make_read_hook(measurements['hidden_states'], layer_idx, step)
                    )
                )
            try:
                with torch.no_grad():
                    model(x)
            finally:
                for h in read_handles:
                    h.remove()

    # ------------------------------------------------------- decode response
    response_token_ids = x[0, prompt_len:].tolist()
    response_tokens    = [tokenizer.decode([tid]) for tid in response_token_ids]

    return measurements, response_tokens
