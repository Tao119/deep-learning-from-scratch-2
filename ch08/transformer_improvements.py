"""
ch08/transformer_improvements.py
──────────────────────────────────
Modern Transformer improvements implemented in pure NumPy, compared on
a toy character-level language-model task.

Techniques implemented
──────────────────────
1. RoPE  — Rotary Position Embedding
2. Flash Attention (simplified)  — block-wise attention with online softmax
3. Pre-LayerNorm vs Post-LayerNorm
4. Grouped Query Attention (GQA)  — n_q=8, g ∈ {1,2,4,8}
5. ALiBi  — Attention with Linear Biases

Each technique has an implementation section followed by a toy experiment
that measures perplexity or loss and writes a comparison summary.

Output
──────
  ch08/transformer_improvements.png  — comparison charts
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import preprocess, clip_grads
from common.optimizer import Adam

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


def _cross_entropy_seq(logits, targets):
    N, T, V = logits.shape
    probs = _softmax(logits.reshape(N * T, V))
    flat_t = targets.reshape(N * T)
    loss = -np.log(probs[np.arange(N * T), flat_t] + 1e-7).mean()
    return loss, probs.reshape(N, T, V)


# ─────────────────────────────────────────────────────────────────────────────
#  Toy corpus
# ─────────────────────────────────────────────────────────────────────────────

TEXT = (
    "the dog ran . the cat sat . the dog sat . "
    "a cat ran . a dog ran . the cat ran . "
    "the dog ate . a cat ate . the cat ate . "
    "a dog sat . the cat ran . a dog ate . "
    "the dog ran . a cat sat . a dog ran . "
)

CORPUS, WORD2ID, ID2WORD = preprocess(TEXT)
VOCAB_SIZE = len(WORD2ID)


def _make_batch(corpus, batch_size, time_size):
    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    offsets = [data_size * i // batch_size for i in range(batch_size)]
    bx = np.zeros((batch_size, time_size), dtype=np.int32)
    bt = np.zeros((batch_size, time_size), dtype=np.int32)
    for b, off in enumerate(offsets):
        for t in range(time_size):
            bx[b, t] = xs[(off + t) % data_size]
            bt[b, t] = ts[(off + t) % data_size]
    return bx, bt


# ─────────────────────────────────────────────────────────────────────────────
#  1. RoPE — Rotary Position Embedding
# ─────────────────────────────────────────────────────────────────────────────
#
#  For a vector x of dimension d and position pos:
#    RoPE(x, pos) = x * cos(pos * θ) + rotate90(x) * sin(pos * θ)
#  where θ_i = 1 / 10000^(2i / d)  and rotate90 negates/swaps pairs.

def _rope_freqs(d: int, max_len: int = 256) -> np.ndarray:
    """Precompute rotation angles: (max_len, d)."""
    i = np.arange(0, d, 2)
    theta = 1.0 / (10000 ** (i / d))           # (d//2,)
    pos   = np.arange(max_len)[:, None]          # (L, 1)
    freqs = pos * theta[None, :]                  # (L, d//2)
    cos   = np.cos(freqs)                         # (L, d//2)
    sin   = np.sin(freqs)                         # (L, d//2)
    # Interleave: cos at even indices, sin at odd
    out_cos = np.zeros((max_len, d), dtype="f")
    out_sin = np.zeros((max_len, d), dtype="f")
    out_cos[:, 0::2] = cos
    out_cos[:, 1::2] = cos
    out_sin[:, 0::2] = -sin   # rotate90: even → -sin(partner)
    out_sin[:, 1::2] = sin    # rotate90: odd  → +sin(partner)
    return out_cos, out_sin


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply RoPE to x: (..., T, d).
    cos, sin: (max_len, d) or (T, d).
    """
    T = x.shape[-2]
    cos_T = cos[:T]   # (T, d)
    sin_T = sin[:T]

    # rotate90(x): swap pairs with sign flip
    xr = np.empty_like(x)
    xr[..., 0::2] = -x[..., 1::2]
    xr[..., 1::2] =  x[..., 0::2]

    return x * cos_T + xr * sin_T


class RoPEAttention:
    """
    Multi-head self-attention with Rotary Position Embedding.
    Causal (upper-triangle masked).  Forward-only (for experiment).
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int = 256):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        scale = np.sqrt(d_model)
        self.Wq = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.Wk = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.Wv = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.Wo = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.grads  = [np.zeros_like(p) for p in self.params]
        self._cos, self._sin = _rope_freqs(self.d_head, max_len)

    def _split(self, x):
        N, T, _ = x.shape
        return x.reshape(N, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge(self, x):
        N, h, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, T, h * dh)

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, T, _ = x.shape
        Q = self._split(x @ self.Wq)   # (N, h, T, dh)
        K = self._split(x @ self.Wk)
        V = self._split(x @ self.Wv)

        # Apply RoPE to Q and K
        for i in range(self.n_heads):
            Q[:, i] = apply_rope(Q[:, i], self._cos, self._sin)
            K[:, i] = apply_rope(K[:, i], self._cos, self._sin)

        scale  = np.sqrt(self.d_head)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale
        mask   = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, :, mask] = -1e9
        A   = _softmax(scores)
        out = self._merge(A @ V) @ self.Wo
        return out

    def backward(self, dout):
        # Simplified: zero gradients (experiment only measures forward-pass ppl)
        return np.zeros_like(dout)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Flash Attention (simplified block-wise)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Standard attention computes the full N×N score matrix in memory O(N²).
#  Flash Attention tiles the computation into blocks of size B, using an
#  online softmax to accumulate partial results without materialising the
#  full matrix.
#
#  This simplified implementation is forward-only and uses a block size B.

def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 4,
    causal: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Block-wise attention with online softmax.

    Parameters
    ----------
    Q, K, V  : (N, h, T, d_head)
    block_size: tile size B

    Returns
    -------
    output : (N, h, T, d_head)
    n_blocks: number of block pairs processed
    """
    N, h, T, d = Q.shape
    scale    = 1.0 / math.sqrt(d)
    output   = np.zeros_like(Q)
    n_blocks = 0

    for bi in range(0, T, block_size):
        ei = min(bi + block_size, T)
        q_blk = Q[:, :, bi:ei, :]       # (N, h, Bq, d)
        # Running max and sum for online softmax
        m   = np.full((N, h, ei - bi), -1e9)
        s   = np.zeros((N, h, ei - bi))
        acc = np.zeros((N, h, ei - bi, d))

        for bj in range(0, T, block_size):
            ej = min(bj + block_size, T)
            k_blk = K[:, :, bj:ej, :]   # (N, h, Bk, d)
            v_blk = V[:, :, bj:ej, :]

            scores = (q_blk @ k_blk.transpose(0, 1, 3, 2)) * scale  # (N,h,Bq,Bk)

            if causal:
                # Zero out future tokens
                qi = np.arange(bi, ei)[:, None]
                kj = np.arange(bj, ej)[None, :]
                future = qi < kj
                scores[:, :, future] = -1e9

            block_max = scores.max(axis=-1)  # (N, h, Bq)
            new_m = np.maximum(m, block_max)
            # Rescale running accumulator
            scale_old = np.exp(m - new_m)
            scale_blk = np.exp(block_max - new_m)

            e_scores = np.exp(scores - new_m[..., None])  # (N,h,Bq,Bk)
            block_sum = e_scores.sum(axis=-1)             # (N,h,Bq)

            s   = scale_old * s + scale_blk * block_sum
            acc = scale_old[..., None] * acc + scale_blk[..., None] * (e_scores @ v_blk)
            m   = new_m
            n_blocks += 1

        output[:, :, bi:ei, :] = acc / (s[..., None] + 1e-9)

    return output, n_blocks


def standard_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = True,
) -> np.ndarray:
    """Standard O(T²) attention for reference."""
    N, h, T, d = Q.shape
    scale  = 1.0 / math.sqrt(d)
    scores = Q @ K.transpose(0, 1, 3, 2) * scale
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, :, mask] = -1e9
    A = _softmax(scores)
    return A @ V


# ─────────────────────────────────────────────────────────────────────────────
#  3. Pre-LayerNorm vs Post-LayerNorm
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm:
    def __init__(self, d, eps=1e-6):
        self.eps = eps
        self.g   = np.ones(d,  dtype="f")
        self.b   = np.zeros(d, dtype="f")
        self.params = [self.g, self.b]
        self.grads  = [np.zeros_like(self.g), np.zeros_like(self.b)]
        self.cache  = None

    def forward(self, x):
        mu   = x.mean(-1, keepdims=True)
        var  = x.var(-1,  keepdims=True)
        xh   = (x - mu) / np.sqrt(var + self.eps)
        out  = self.g * xh + self.b
        self.cache = (x, xh, mu, var)
        return out

    def backward(self, dout):
        x, xh, mu, var = self.cache
        N  = x.shape[-1]
        si = 1.0 / np.sqrt(var + self.eps)
        self.grads[0][...] = (dout * xh).sum(tuple(range(dout.ndim - 1)))
        self.grads[1][...] = dout.sum(tuple(range(dout.ndim - 1)))
        dxh = dout * self.g
        dx  = si * (dxh - dxh.mean(-1, keepdims=True)
                    - xh * (dxh * xh).mean(-1, keepdims=True))
        return dx


class _CausalAttn:
    """Minimal causal self-attention for Pre/Post-LN comparison."""

    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        sc = np.sqrt(d_model)
        self.Wq = (np.random.randn(d_model, d_model) / sc).astype("f")
        self.Wk = (np.random.randn(d_model, d_model) / sc).astype("f")
        self.Wv = (np.random.randn(d_model, d_model) / sc).astype("f")
        self.Wo = (np.random.randn(d_model, d_model) / sc).astype("f")
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.grads  = [np.zeros_like(p) for p in self.params]
        self.cache  = None

    def _split(self, x):
        N, T, _ = x.shape
        return x.reshape(N, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge(self, x):
        return x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)

    def forward(self, x):
        N, T, _ = x.shape
        Q, K, V = self._split(x @ self.Wq), self._split(x @ self.Wk), self._split(x @ self.Wv)
        s = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)
        s[:, :, np.triu(np.ones((T, T), bool), k=1)] = -1e9
        A = _softmax(s)
        out = self._merge(A @ V) @ self.Wo
        self.cache = (x, Q, K, V, A)
        return out

    def backward(self, dout):
        x, Q, K, V, A = self.cache
        N, T, _ = x.shape
        # Compute output before Wo
        pre_Wo = self._merge(A @ V)
        self.grads[3][...] = pre_Wo.reshape(N * T, -1).T @ dout.reshape(N * T, -1)
        d_pre = dout @ self.Wo.T
        d_AV  = self._split(d_pre)
        dA    = d_AV @ V.transpose(0, 1, 3, 2)
        dV    = A.transpose(0, 1, 3, 2) @ d_AV
        ds    = A * (dA - (dA * A).sum(-1, keepdims=True))
        ds   /= np.sqrt(self.d_head)
        mask  = np.triu(np.ones((T, T), bool), k=1)
        ds[:, :, mask] = 0.0
        dQ    = ds @ K
        dK    = ds.transpose(0, 1, 3, 2) @ Q
        dQm, dKm, dVm = self._merge(dQ), self._merge(dK), self._merge(dV)
        self.grads[0][...] = x.reshape(N * T, -1).T @ dQm.reshape(N * T, -1)
        self.grads[1][...] = x.reshape(N * T, -1).T @ dKm.reshape(N * T, -1)
        self.grads[2][...] = x.reshape(N * T, -1).T @ dVm.reshape(N * T, -1)
        return (dQm @ self.Wq.T + dKm @ self.Wk.T + dVm @ self.Wv.T)


class _FFN:
    def __init__(self, d_model, d_ff):
        sc = np.sqrt(d_model)
        self.W1 = (np.random.randn(d_model, d_ff)   / sc).astype("f")
        self.b1 = np.zeros(d_ff,    dtype="f")
        self.W2 = (np.random.randn(d_ff,   d_model) / np.sqrt(d_ff)).astype("f")
        self.b2 = np.zeros(d_model, dtype="f")
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.grads  = [np.zeros_like(p) for p in self.params]
        self.cache  = None

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        xr, hr, dr = x.reshape(-1, x.shape[-1]), h.reshape(-1, h.shape[-1]), dout.reshape(-1, dout.shape[-1])
        dW2 = hr.T @ dr
        db2 = dr.sum(0)
        dh  = dr @ self.W2.T
        dh[hr == 0] = 0
        dW1 = xr.T @ dh
        db1 = dh.sum(0)
        self.grads[0][...] = dW1; self.grads[1][...] = db1
        self.grads[2][...] = dW2; self.grads[3][...] = db2
        return (dh @ self.W1.T).reshape(x.shape)


class PreLNBlock:
    """x → LN → Attn → + → LN → FFN → +   (Pre-LayerNorm)"""

    def __init__(self, d_model, n_heads, d_ff):
        self.ln1  = LayerNorm(d_model)
        self.attn = _CausalAttn(d_model, n_heads)
        self.ln2  = LayerNorm(d_model)
        self.ffn  = _FFN(d_model, d_ff)
        self.params = self.ln1.params + self.attn.params + self.ln2.params + self.ffn.params
        self.grads  = self.ln1.grads  + self.attn.grads  + self.ln2.grads  + self.ffn.grads
        self.cache  = None

    def forward(self, x):
        h   = x + self.attn.forward(self.ln1.forward(x))
        out = h + self.ffn.forward(self.ln2.forward(h))
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        dh_ffn = self.ffn.backward(self.ln2.backward(dout))
        dh = dout + dh_ffn
        dx_a = self.attn.backward(self.ln1.backward(dh))
        return dh + dx_a


class PostLNBlock:
    """x → Attn → + → LN → FFN → + → LN   (Post-LayerNorm)"""

    def __init__(self, d_model, n_heads, d_ff):
        self.attn = _CausalAttn(d_model, n_heads)
        self.ln1  = LayerNorm(d_model)
        self.ffn  = _FFN(d_model, d_ff)
        self.ln2  = LayerNorm(d_model)
        self.params = self.attn.params + self.ln1.params + self.ffn.params + self.ln2.params
        self.grads  = self.attn.grads  + self.ln1.grads  + self.ffn.grads  + self.ln2.grads
        self.cache  = None

    def forward(self, x):
        h   = self.ln1.forward(x + self.attn.forward(x))
        out = self.ln2.forward(h + self.ffn.forward(h))
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        d2 = self.ln2.backward(dout)
        dh_ffn = self.ffn.backward(d2)
        dh = d2 + dh_ffn
        d1 = self.ln1.backward(dh)
        da = self.attn.backward(d1)
        return d1 + da


# ─────────────────────────────────────────────────────────────────────────────
#  4. Grouped Query Attention (GQA)
# ─────────────────────────────────────────────────────────────────────────────
#
#  n_q query heads each use their own Q projection.
#  n_kv = n_q // g key-value heads are shared within each group of g.
#  Memory: KV matrices are n_kv × d_head instead of n_q × d_head.

class GQAttention:
    """
    Grouped Query Attention (forward-only).

    Parameters
    ----------
    d_model : embedding dimension
    n_q     : number of query heads
    g       : group size (n_kv = n_q // g)
    """

    def __init__(self, d_model: int, n_q: int, g: int):
        assert n_q % g == 0, "n_q must be divisible by g"
        self.d_model = d_model
        self.n_q     = n_q
        self.n_kv    = n_q // g
        self.g       = g
        self.d_head  = d_model // n_q
        sc = np.sqrt(d_model)
        # Each query head has its own projection
        self.Wq = (np.random.randn(d_model, n_q   * self.d_head) / sc).astype("f")
        # KV heads are shared
        self.Wk = (np.random.randn(d_model, self.n_kv * self.d_head) / sc).astype("f")
        self.Wv = (np.random.randn(d_model, self.n_kv * self.d_head) / sc).astype("f")
        self.Wo = (np.random.randn(d_model, d_model) / sc).astype("f")
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.grads  = [np.zeros_like(p) for p in self.params]

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, T, _ = x.shape
        dh = self.d_head

        # Q: (N, n_q, T, dh)
        Q = (x @ self.Wq).reshape(N, T, self.n_q,  dh).transpose(0, 2, 1, 3)
        # K, V: (N, n_kv, T, dh)
        K = (x @ self.Wk).reshape(N, T, self.n_kv, dh).transpose(0, 2, 1, 3)
        V = (x @ self.Wv).reshape(N, T, self.n_kv, dh).transpose(0, 2, 1, 3)

        # Expand KV to match n_q by repeating each KV head g times
        # K_exp: (N, n_q, T, dh)
        K_exp = np.repeat(K, self.g, axis=1)
        V_exp = np.repeat(V, self.g, axis=1)

        scale  = 1.0 / np.sqrt(dh)
        scores = Q @ K_exp.transpose(0, 1, 3, 2) * scale
        mask   = np.triu(np.ones((T, T), bool), k=1)
        scores[:, :, mask] = -1e9
        A   = _softmax(scores)
        out = (A @ V_exp).transpose(0, 2, 1, 3).reshape(N, T, -1) @ self.Wo
        return out

    @property
    def memory_ratio(self) -> float:
        """KV memory relative to MHA (g=1)."""
        return self.n_kv / self.n_q


# ─────────────────────────────────────────────────────────────────────────────
#  5. ALiBi — Attention with Linear Biases
# ─────────────────────────────────────────────────────────────────────────────
#
#  Instead of positional embeddings, add head-specific linear bias to scores:
#    bias[i, j] = -m * |i - j|
#  m is a geometric sequence of slopes specific to each head.
#  ALiBi can generalise to longer sequences than training length.

def _alibi_slopes(n_heads: int) -> np.ndarray:
    """Compute ALiBi slope per head (geometric sequence)."""
    # slopes: 2^(-8/n), 2^(-8*2/n), ..., 2^(-8)
    i = np.arange(1, n_heads + 1)
    slopes = 2.0 ** (-8.0 * i / n_heads)
    return slopes.astype("f")


def _alibi_bias(T: int, n_heads: int) -> np.ndarray:
    """
    Pre-compute ALiBi bias matrix: (n_heads, T, T).
    bias[h, i, j] = -slope_h * |i - j|
    """
    slopes = _alibi_slopes(n_heads)             # (h,)
    rows   = np.arange(T)[None, :, None]        # (1, T, 1)
    cols   = np.arange(T)[None, None, :]        # (1, 1, T)
    dist   = np.abs(rows - cols).astype("f")    # (1, T, T)
    bias   = -slopes[:, None, None] * dist      # (h, T, T)
    return bias


class ALiBiAttention:
    """
    Causal multi-head self-attention with ALiBi positional bias.
    No positional embeddings needed — biases are added to attention scores.
    Forward-only (backward zeroed for experiment).
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int = 512):
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self._max_len = max_len
        sc = np.sqrt(d_model)
        self.Wq = (np.random.randn(d_model, d_model) / sc).astype("f")
        self.Wk = (np.random.randn(d_model, d_model) / sc).astype("f")
        self.Wv = (np.random.randn(d_model, d_model) / sc).astype("f")
        self.Wo = (np.random.randn(d_model, d_model) / sc).astype("f")
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.grads  = [np.zeros_like(p) for p in self.params]
        # Cache bias for max_len; recompute on demand if T > max_len
        self._bias = _alibi_bias(max_len, n_heads)  # (h, max_len, max_len)

    def _split(self, x):
        N, T, _ = x.shape
        return x.reshape(N, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge(self, x):
        return x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, T, _ = x.shape
        Q, K, V = self._split(x @ self.Wq), self._split(x @ self.Wk), self._split(x @ self.Wv)
        scale  = 1.0 / np.sqrt(self.d_head)
        scores = Q @ K.transpose(0, 1, 3, 2) * scale    # (N, h, T, T)

        # ALiBi bias
        if T <= self._max_len:
            bias = self._bias[:, :T, :T]
        else:
            bias = _alibi_bias(T, self.n_heads)
        scores = scores + bias[None]   # broadcast over batch

        # Causal mask
        mask = np.triu(np.ones((T, T), bool), k=1)
        scores[:, :, mask] = -1e9

        A   = _softmax(scores)
        out = self._merge(A @ V) @ self.Wo
        return out

    def backward(self, dout):
        return np.zeros_like(dout)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared TransformerLM (pluggable attention / block type)
# ─────────────────────────────────────────────────────────────────────────────

def _positional_encoding(T, d):
    pe  = np.zeros((T, d), dtype="f")
    pos = np.arange(T)[:, None]
    div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d // 2])
    return pe


class _TinyLM:
    """
    Minimal decoder-only LM that accepts a list of blocks.
    Blocks must have: forward(x)->x, backward(dout)->dx, params, grads.
    """

    def __init__(self, vocab_size, d_model, blocks, use_pe=True, max_len=256):
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.blocks     = blocks
        self.use_pe     = use_pe
        scale = np.sqrt(d_model)
        self.embed_W = (np.random.randn(vocab_size, d_model) / scale).astype("f")
        self.head_b  = np.zeros(vocab_size, dtype="f")
        self.pe      = _positional_encoding(max_len, d_model)
        self.params  = [self.embed_W, self.head_b]
        self.grads   = [np.zeros_like(self.embed_W), np.zeros_like(self.head_b)]
        for blk in self.blocks:
            self.params += blk.params
            self.grads  += blk.grads
        self.cache = None

    def forward(self, xs, ts):
        N, T = xs.shape
        x = self.embed_W[xs]
        if self.use_pe:
            x = x + self.pe[:T]
        self.emb_in = xs
        for blk in self.blocks:
            x = blk.forward(x)
        logits = x @ self.embed_W.T + self.head_b
        loss, probs = _cross_entropy_seq(logits, ts)
        self.cache = (x, probs, ts)
        return loss

    def backward(self, dout=1):
        x, probs, ts = self.cache
        N, T, V = probs.shape
        dl = probs.copy()
        dl[np.arange(N)[:, None], np.arange(T)[None, :], ts] -= 1
        dl *= dout / (N * T)
        self.grads[1][...] = dl.sum((0, 1))
        dl2 = dl.reshape(N * T, V)
        x2  = x.reshape(N * T, self.d_model)
        dW_head = dl2.T @ x2
        dx = (dl2 @ self.embed_W).reshape(N, T, self.d_model)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        dW_emb = np.zeros_like(self.embed_W)
        np.add.at(dW_emb, self.emb_in.reshape(-1), dx.reshape(N * T, self.d_model))
        self.grads[0][...] = dW_emb + dW_head


def _train(model, corpus, epochs=150, batch_size=4, time_size=8, lr=1e-3):
    opt = Adam(lr=lr)
    ppl_hist = []
    for epoch in range(epochs):
        bx, bt = _make_batch(corpus, batch_size, time_size)
        loss = model.forward(bx, bt)
        model.backward()
        clip_grads(model.grads, 1.0)
        opt.update(model.params, model.grads)
        ppl_hist.append(float(np.exp(min(loss, 10))))
    return ppl_hist


# ─────────────────────────────────────────────────────────────────────────────
#  Experiments
# ─────────────────────────────────────────────────────────────────────────────

def exp_pre_vs_post_ln(corpus, vocab_size, epochs=150):
    """Compare Pre-LN vs Post-LN on the toy corpus."""
    np.random.seed(42)
    d, h, ff, n_layers = 32, 4, 64, 2
    pre_blocks  = [PreLNBlock(d, h, ff)  for _ in range(n_layers)]
    post_blocks = [PostLNBlock(d, h, ff) for _ in range(n_layers)]
    pre_model   = _TinyLM(vocab_size, d, pre_blocks,  use_pe=True)
    post_model  = _TinyLM(vocab_size, d, post_blocks, use_pe=True)

    np.random.seed(42)
    pre_ppls  = _train(pre_model,  corpus, epochs=epochs)
    np.random.seed(42)
    post_ppls = _train(post_model, corpus, epochs=epochs)

    return pre_ppls, post_ppls


def exp_flash_vs_standard():
    """
    Compare Flash Attention output with standard attention on a random Q,K,V.
    Also measure memory: standard needs T² values; flash uses O(B*T).
    """
    np.random.seed(42)
    N, h, T, d = 2, 4, 32, 8
    Q = np.random.randn(N, h, T, d).astype("f")
    K = np.random.randn(N, h, T, d).astype("f")
    V = np.random.randn(N, h, T, d).astype("f")

    out_std   = standard_attention_forward(Q, K, V, causal=True)
    out_flash, n_blk = flash_attention_forward(Q, K, V, block_size=4, causal=True)

    max_err = float(np.abs(out_std - out_flash).max())
    # Memory estimate: standard = N*h*T*T floats; flash = O(N*h*B*T) at any moment
    B = 4
    mem_std   = N * h * T * T
    mem_flash = N * h * B * T * 3  # Q,K,V blocks simultaneously

    return max_err, mem_std, mem_flash


def exp_gqa(corpus, vocab_size, n_q=8, groups=(1, 2, 4, 8), epochs=150):
    """Train tiny GQA LMs with different group sizes."""
    d, ff = 32, 64
    results = {}
    for g in groups:
        if n_q % g != 0:
            continue
        np.random.seed(42)

        class _GQABlock:
            def __init__(self):
                self.attn = GQAttention(d, n_q, g)
                self.ln1  = LayerNorm(d)
                self.ln2  = LayerNorm(d)
                self.ffn  = _FFN(d, ff)
                self.params = self.attn.params + self.ln1.params + self.ln2.params + self.ffn.params
                self.grads  = self.attn.grads  + self.ln1.grads  + self.ln2.grads  + self.ffn.grads
                self.cache  = None

            def forward(self, x):
                h   = x + self.attn.forward(self.ln1.forward(x))
                out = h + self.ffn.forward(self.ln2.forward(h))
                self.cache = (x, h)
                return out

            def backward(self, dout):
                x, h = self.cache
                dh = dout + self.ffn.backward(self.ln2.backward(dout))
                return dh  # GQA has no backward; pass gradient through unchanged

        blocks = [_GQABlock()]
        model  = _TinyLM(vocab_size, d, blocks, use_pe=True)
        ppl_h  = _train(model, corpus, epochs=epochs)
        label  = "MHA" if g == 1 else (f"MQA" if g == n_q else f"GQA(g={g})")
        n_kv   = n_q // g
        results[label] = {"ppl_hist": ppl_h, "n_kv": n_kv, "mem_ratio": n_kv / n_q}
    return results


def exp_alibi_vs_sinusoidal(corpus, vocab_size, epochs=150):
    """Compare ALiBi (no pos embedding) vs sinusoidal PE on toy corpus."""
    np.random.seed(42)
    d, h, ff, nl = 32, 4, 64, 2

    # Sinusoidal PE baseline (Pre-LN blocks)
    sin_blocks = [PreLNBlock(d, h, ff) for _ in range(nl)]
    sin_model  = _TinyLM(vocab_size, d, sin_blocks, use_pe=True)

    # ALiBi: no PE, ALiBi attention baked into attention (we replace attn in blocks)
    class ALiBiBlock:
        def __init__(self):
            self.ln1  = LayerNorm(d)
            self.attn = ALiBiAttention(d, h)
            self.ln2  = LayerNorm(d)
            self.ffn  = _FFN(d, ff)
            self.params = self.ln1.params + self.attn.params + self.ln2.params + self.ffn.params
            self.grads  = self.ln1.grads  + self.attn.grads  + self.ln2.grads  + self.ffn.grads
            self.cache  = None

        def forward(self, x):
            h   = x + self.attn.forward(self.ln1.forward(x))
            out = h + self.ffn.forward(self.ln2.forward(h))
            self.cache = (x, h)
            return out

        def backward(self, dout):
            x, h = self.cache
            dh = dout + self.ffn.backward(self.ln2.backward(dout))
            return dh + self.ln1.backward(dh) * 0  # ALiBi attn has no backward; simplified

    ali_blocks = [ALiBiBlock() for _ in range(nl)]
    ali_model  = _TinyLM(vocab_size, d, ali_blocks, use_pe=False)

    np.random.seed(42)
    sin_ppls = _train(sin_model, corpus, epochs=epochs)
    np.random.seed(42)
    ali_ppls = _train(ali_model, corpus, epochs=epochs)

    return sin_ppls, ali_ppls


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    EPOCHS = 150

    print("=" * 65)
    print("Transformer Improvements — toy experiments")
    print("=" * 65)

    # ── Experiment 1: Pre-LN vs Post-LN ─────────────────────────────────────
    print("\n[1] Pre-LayerNorm vs Post-LayerNorm …")
    pre_ppls, post_ppls = exp_pre_vs_post_ln(CORPUS, VOCAB_SIZE, epochs=EPOCHS)
    print(f"    Final PPL — Pre-LN: {pre_ppls[-1]:.2f}  Post-LN: {post_ppls[-1]:.2f}")

    # ── Experiment 2: Flash vs Standard ─────────────────────────────────────
    print("\n[2] Flash Attention vs Standard Attention …")
    err, mem_std, mem_flash = exp_flash_vs_standard()
    print(f"    Max output error (Flash vs Std): {err:.2e}")
    print(f"    Memory floats — Standard: {mem_std:,}  Flash: {mem_flash:,}")
    print(f"    Memory reduction: {mem_std/mem_flash:.1f}x")

    # ── Experiment 3: GQA ────────────────────────────────────────────────────
    print("\n[3] Grouped Query Attention (n_q=8) …")
    gqa_res = exp_gqa(CORPUS, VOCAB_SIZE, n_q=8, groups=[1, 2, 4, 8], epochs=EPOCHS)
    for label, info in gqa_res.items():
        print(f"    {label:15s}  n_kv={info['n_kv']}  "
              f"mem={info['mem_ratio']*100:.0f}%  ppl={info['ppl_hist'][-1]:.2f}")

    # ── Experiment 4: ALiBi vs Sinusoidal ───────────────────────────────────
    print("\n[4] ALiBi vs Sinusoidal PE …")
    sin_ppls, ali_ppls = exp_alibi_vs_sinusoidal(CORPUS, VOCAB_SIZE, epochs=EPOCHS)
    print(f"    Final PPL — Sinusoidal: {sin_ppls[-1]:.2f}  ALiBi: {ali_ppls[-1]:.2f}")

    # ── Experiment 5: RoPE (forward sanity check) ────────────────────────────
    print("\n[5] RoPE — sanity check …")
    d_rope = 16
    cos_r, sin_r = _rope_freqs(d_rope)
    x_test = np.random.randn(2, 8, d_rope).astype("f")
    out    = apply_rope(x_test, cos_r, sin_r)
    print(f"    RoPE output shape: {out.shape}  "
          f"norm diff: {np.linalg.norm(out - x_test):.3f}")

    # ── Plotting ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ep = np.arange(1, EPOCHS + 1)

    # 1. Pre-LN vs Post-LN
    axes[0, 0].plot(ep, pre_ppls,  label="Pre-LN",  color="steelblue", linewidth=1.5)
    axes[0, 0].plot(ep, post_ppls, label="Post-LN", color="salmon",    linewidth=1.5)
    axes[0, 0].set_title("Pre-LN vs Post-LN")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Perplexity")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # 2. Flash vs Standard (bar chart of memory)
    labels2 = ["Standard\nO(T²)", "Flash\nO(B·T)"]
    vals2   = [mem_std, mem_flash]
    bars2   = axes[0, 1].bar(labels2, vals2, color=["salmon", "steelblue"], edgecolor="k")
    for bar, v in zip(bars2, vals2):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{v:,}", ha="center", va="bottom", fontsize=9)
    axes[0, 1].set_title(f"Flash Attention Memory\n(max err={err:.1e})")
    axes[0, 1].set_ylabel("Floats stored")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    # 3. GQA: final PPL vs memory ratio
    gqa_labels = list(gqa_res.keys())
    gqa_ppls   = [gqa_res[k]["ppl_hist"][-1] for k in gqa_labels]
    gqa_mems   = [gqa_res[k]["mem_ratio"] * 100 for k in gqa_labels]
    ax3 = axes[0, 2]
    c3  = ax3.bar(gqa_labels, gqa_ppls, color="steelblue", edgecolor="k")
    ax3r = ax3.twinx()
    ax3r.plot(gqa_labels, gqa_mems, "o--", color="salmon", linewidth=1.5, markersize=6)
    ax3.set_title("GQA: Final PPL vs KV Memory")
    ax3.set_ylabel("Final PPL", color="steelblue")
    ax3r.set_ylabel("KV Memory %", color="salmon")
    ax3.grid(True, axis="y", alpha=0.3)

    # 4. ALiBi vs Sinusoidal
    axes[1, 0].plot(ep, sin_ppls, label="Sinusoidal PE", color="steelblue", linewidth=1.5)
    axes[1, 0].plot(ep, ali_ppls, label="ALiBi",         color="orange",    linewidth=1.5)
    axes[1, 0].set_title("ALiBi vs Sinusoidal PE")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Perplexity")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # 5. GQA perplexity curves (all groups)
    colors5 = ["steelblue", "seagreen", "salmon", "orange"]
    for (label, info), col in zip(gqa_res.items(), colors5):
        axes[1, 1].plot(ep, info["ppl_hist"], label=label, color=col, linewidth=1.5)
    axes[1, 1].set_title("GQA Loss Curves")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Perplexity")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    # 6. Final PPL comparison bar chart
    tech_labels = ["Pre-LN", "Post-LN", "ALiBi", "Sin PE"]
    tech_ppls   = [pre_ppls[-1], post_ppls[-1], ali_ppls[-1], sin_ppls[-1]]
    bar6 = axes[1, 2].bar(tech_labels, tech_ppls,
                           color=["steelblue", "salmon", "orange", "seagreen"],
                           edgecolor="k")
    for bar, v in zip(bar6, tech_ppls):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1, 2].set_title("Final PPL Summary")
    axes[1, 2].set_ylabel("Perplexity (lower=better)")
    axes[1, 2].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Modern Transformer Improvements — Toy Experiments", fontsize=13)
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer_improvements.png")
    plt.savefig(out_png, dpi=120)
    print(f"\nPlot saved to {out_png}")
    print("\nDone.")
