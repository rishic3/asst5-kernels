import torch
import sys
sys.path.insert(0, '..')

from reference import ref_kernel, generate_input
from submission import custom_kernel

# Generate test input
data = generate_input(batch_size=256, in_features=2048, hidden_size=4096, seed=8846, seq=64)
x, W, V, b, c, beta = data

print(f"Input shapes:")
print(f"  x: {x.shape}, dtype: {x.dtype}")
print(f"  W: {W.shape}, dtype: {W.dtype}")
print(f"  V: {V.shape}, dtype: {V.dtype}")
print(f"  b: {b.shape}, dtype: {b.dtype}")
print(f"  c: {c.shape}, dtype: {c.dtype}")
print(f"  beta: {beta}")

# Get reference output
ref_out = ref_kernel(data)
print(f"\nReference output: {ref_out.shape}, dtype: {ref_out.dtype}")

# Get custom output
custom_out = custom_kernel(data)
print(f"Custom output: {custom_out.shape}, dtype: {custom_out.dtype}")

# Compare
print(f"\nComparison:")
print(f"  Max abs diff: {(ref_out - custom_out).abs().max().item()}")
print(f"  Mean abs diff: {(ref_out - custom_out).abs().mean().item()}")
print(f"  Max ref value: {ref_out.abs().max().item()}")
print(f"  Max custom value: {custom_out.abs().max().item()}")

# Check relative error
rel_diff = (ref_out - custom_out).abs() / (ref_out.abs() + 1e-8)
print(f"  Max relative diff: {rel_diff.max().item()}")
print(f"  Mean relative diff: {rel_diff.mean().item()}")

# Check tolerance used in check_implementation
rtol, atol = 1e-2, 1e-2
is_close = torch.allclose(custom_out, ref_out, rtol=rtol, atol=atol)
print(f"\n  torch.allclose (rtol={rtol}, atol={atol}): {is_close}")

# Find where differences are largest
diff = (ref_out - custom_out).abs()
flat_idx = diff.argmax().item()
idx = []
shape = ref_out.shape
for dim in reversed(shape):
    idx.insert(0, flat_idx % dim)
    flat_idx //= dim
print(f"\n  Largest diff at index {tuple(idx)}:")
print(f"    ref value: {ref_out[tuple(idx)].item()}")
print(f"    custom value: {custom_out[tuple(idx)].item()}")

# Let's also check intermediate values - compute what we expect
print("\n\nDebugging intermediate values...")
x_f32 = x.float()
W_f32 = W.float()
V_f32 = V.float()
b_f32 = b.float()
c_f32 = c.float()

# Reference computation
gate_ref = x_f32 @ W_f32 + b_f32
value_ref = x_f32 @ V_f32 + c_f32
swish_gate_ref = gate_ref * torch.sigmoid(beta * gate_ref)
expected = swish_gate_ref * value_ref

print(f"Expected (recomputed): max={expected.abs().max().item():.6f}")
print(f"ref_out: max={ref_out.abs().max().item():.6f}")
print(f"Match expected vs ref_out: {torch.allclose(expected, ref_out.float(), rtol=1e-3, atol=1e-3)}")

