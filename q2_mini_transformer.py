"""
Mini Transformer Encoder demo
- Small set of 10 sentences
- Tokenize (whitespace), embed, add sinusoidal positional encoding
- Single encoder block with multi-head self-attention (2 or 4 heads)
- Prints input tokens, final contextual embeddings, and saves an attention heatmap
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1) Small dataset (10 sentences)
# -----------------------------
sentences = [
    "Sherlock Holmes loved logic.",
    "Watson recorded the adventures.",
    "The detective examined clues carefully.",
    "A case begins with a call.",
    "They walked down Baker Street.",
    "A mystery often hides in plain sight.",
    "He lit his pipe and thought.",
    "The client explained the strange events.",
    "Holmes asked a few precise questions.",
    "They solved the puzzle together."
]

# -----------------------------
# 2) Simple tokenizer (whitespace)
# -----------------------------
def tokenize(sent):
    return sent.strip().split()

tokenized = [tokenize(s) for s in sentences]
vocab = {}
for sent in tokenized:
    for w in sent:
        if w not in vocab:
            vocab[w] = len(vocab)
# add special PAD token (not needed heavily here)
vocab['<PAD>'] = len(vocab)

itos = {i:w for w, i in vocab.items()}

print("Vocabulary (tokens):", list(vocab.keys()))
max_len = max(len(t) for t in tokenized)
print("Max tokens in a sentence:", max_len)

# -----------------------------
# 3) Embedding + Positional Encoding
# -----------------------------
d_model = 64
num_heads = 4  # choose 2 or 4
ff_dim = 128

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x shape: (B, T, D)
        return x + self.pe[:, :x.size(1), :]

# -----------------------------
# 4) Multi-head self-attention
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=64, heads=4):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.size()
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape for heads
        q = q.view(B, T, self.heads, self.d_k).transpose(1,2)  # (B, heads, T, d_k)
        k = k.view(B, T, self.heads, self.d_k).transpose(1,2)
        v = v.view(B, T, self.heads, self.d_k).transpose(1,2)
        # scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, heads, T, T)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, T, d_k)
        out = out.transpose(1,2).contiguous().view(B, T, D)  # (B, T, D)
        return self.out(out), attn  # return attention for inspection

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=64, heads=4, ff_dim=128):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.mha(x)
        x = self.norm1(x + attn_out)
        f = self.ff(x)
        x = self.norm2(x + f)
        return x, attn_weights

# -----------------------------
# 5) Build model components
# -----------------------------
embed = nn.Embedding(len(vocab), d_model)
pos_enc = PositionalEncoding(d_model, max_len=max_len)
encoder_block = TransformerEncoderBlock(d_model=d_model, heads=num_heads, ff_dim=ff_dim)

# Prepare batch (B = number of sentences)
batch_tokens = []
token_masks = []
for sent in tokenized:
    ids = [vocab[w] for w in sent]
    # pad to max_len
    ids = ids + [vocab['<PAD>']] * (max_len - len(ids))
    batch_tokens.append(ids)
    token_masks.append([1]*len(sent) + [0]*(max_len - len(sent)))

batch_tokens = torch.tensor(batch_tokens)  # (B, T)
token_masks = torch.tensor(token_masks)    # (B, T)

# -----------------------------
# 6) Forward pass through encoder
# -----------------------------
x = embed(batch_tokens)  # (B, T, D)
x = pos_enc(x)
print("\nInput tokens (first sentence):", tokenized[0])
print("Embedded input shape:", x.shape)

out, attn_weights = encoder_block(x)  # out: (B, T, D), attn_weights: (B, heads, T, T)
print("\nFinal contextual embeddings shape:", out.shape)

# print final contextual embeddings (for the first sentence, first few tokens)
print("\nFinal embeddings (sentence 0, token 0..min(5,T)):")
for i in range(min(5, out.size(1))):
    vec = out[0, i].detach().numpy()
    print(f" token {i} ({itos[batch_tokens[0,i].item()]}): mean={vec.mean():.4f}, std={vec.std():.4f}")

# -----------------------------
# 7) Visualize attention heatmap for sentence 0 and head 0
# -----------------------------
head = 0
sent_idx = 0
attn_matrix = attn_weights[sent_idx, head].detach().numpy()  # (T,T)
tokens = [itos[id_.item()] for id_ in batch_tokens[sent_idx]]

plt.figure(figsize=(8,6))
plt.imshow(attn_matrix, cmap='viridis')
plt.colorbar()
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.title(f"Attention heatmap (sentence {sent_idx}, head {head})")
plt.tight_layout()
plt.savefig("q2_attention_heatmap.png")
print("\nSaved attention heatmap to q2_attention_heatmap.png")

# Also print a small slice of attention matrix for inspection
print("\nAttention matrix (first 8 tokens):")
print(np.round(attn_matrix[:8,:8], 3))

# End
print("\nQ2 done.")
