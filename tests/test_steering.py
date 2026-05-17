import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch
from steering import apply_steering

# --- build synthetic test case ---
d_model = 4096
batch, seq_len = 1, 10

# random hidden state
torch.manual_seed(0)
h = torch.randn(batch, seq_len, d_model)

# construct v with a KNOWN component in h so norm reduction is guaranteed visible
# inject v into h so we know the projection will do something measurable
v_raw = torch.randn(d_model)
v = v_raw / v_raw.norm()                        # unit vector
h = h + 5.0 * v.unsqueeze(0).unsqueeze(0)      # inject: h now has component=5 along v

print(f"=== Input ===")
print(f"h shape:    {h.shape}")
print(f"v shape:    {v.shape}")
print(f"v norm:     {v.norm():.6f}")
proj_before = torch.einsum('bsd,d->bs', h.float(), v.float())
print(f"proj before (mean abs): {proj_before.abs().mean():.4f}  (should be ~5.0)")

# --- call the function ---
result = apply_steering(h, v, debug=True)

# --- external verification ---
print(f"\n=== External Verification ===")
v_unit = (v.float() / v.float().norm())
proj_after = torch.einsum('bsd,d->bs', result.float(), v_unit)
print(f"proj after  (mean abs): {proj_after.abs().mean():.6f}  (should be ~0)")
print(f"norm in:  {h.float().norm():.4f}")
print(f"norm out: {result.float().norm():.4f}")
print(f"norm reduction: {(h.float().norm() - result.float().norm()):.4f}  (should be > 0)")
