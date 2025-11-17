"""
Scaled dot-product attention implementation + softmax stability check.

- Implements attention(Q,K,V)
- Tests with random Q,K,V
- Compares softmax before scaling vs after scaling to demonstrate numerical behavior
"""

import torch
import torch.nn.functional as F
import math
import numpy as np

def scaled_dot_product_attention(Q, K, V, do_scale=True):
    """
    Q: (B, T_q, d)
    K: (B, T_k, d)
    V: (B, T_k, d_v)
    returns: output (B, T_q, d_v), weights (B, T_q, T_k)
    """
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, T_q, T_k)
    if do_scale:
        scores = scores / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights

# -----------------------------
# Test with random inputs
# -----------------------------
B = 1
T = 5
d = 64
torch.manual_seed(0)

Q = torch.randn(B, T, d) * 5.0  # amplify to create larger dot products
K = torch.randn(B, T, d) * 5.0
V = torch.randn(B, T, d)

# Without scaling
out_ns, w_ns = scaled_dot_product_attention(Q, K, V, do_scale=False)
# With scaling
out_s, w_s = scaled_dot_product_attention(Q, K, V, do_scale=True)

print("Attention weights WITHOUT scaling (shape):", w_ns.shape)
print("Attention weights WITH scaling (shape):", w_s.shape)

# Print weight matrices for a single query (first row)
print("\nWeights WITHOUT scaling (first query row):")
print(np.round(w_ns[0,0].detach().numpy(), 4))

print("\nWeights WITH scaling (first query row):")
print(np.round(w_s[0,0].detach().numpy(), 4))

# Show that without scaling, distribution tends to be very peaked (one near-1), with scaling it's smoother
print("\nSum of weights (sanity):", w_ns[0,0].sum().item(), w_s[0,0].sum().item())

# Print outputs shapes and a sample output vector
print("\nOutput shape WITHOUT scaling:", out_ns.shape)
print("Output shape WITH scaling:", out_s.shape)
print("\nSample output vector (first token) WITHOUT scaling (first 6 dims):")
print(np.round(out_ns[0,0,:6].detach().numpy(), 4))
print("\nSample output vector (first token) WITH scaling (first 6 dims):")
print(np.round(out_s[0,0,:6].detach().numpy(), 4))

# Softmax stability: show scores range before softmax
scores_ns = torch.matmul(Q, K.transpose(-2,-1))
scores_s = scores_ns / math.sqrt(d)
print("\nScores range WITHOUT scaling: min, max =", scores_ns.min().item(), scores_ns.max().item())
print("Scores range WITH scaling:    min, max =", scores_s.min().item(), scores_s.max().item())

print("\nQ3 done.")
