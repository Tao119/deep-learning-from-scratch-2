"""
ch08/sparse_attention.py
─────────────────────────
Implement and compare four attention sparsity patterns on a toy LM task.

Patterns
────────
1. Full          — standard O(n²) dense attention
2. Local (sliding window) — each token attends to its ±w neighbours only
3. Strided       — token i attends to i, i-1, …, and also i-s, i-2s, …
4. BigBird-style — random + local + global tokens

For each pattern:
  - Build the boolean attention mask
  - Train a tiny Transformer LM (same toy corpus)
  - Measure final perplexity and estimated FLOPS

Output
──────
  ch08/sparse_attention_comparison.png
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import preprocess, clip_grads
from common.optimizer import Adam

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
#  Attention mask factories
# ─────────────────────────────────────────────────────────────────────────────
#
#  Each factory returns a boolean mask of shape (T, T).
#  mask[i, j] = True  → token i CANNOT attend to j  (will be set to -inf)
#  mask[i, j] = False → token i CAN attend to j

def make_full_mask(T: int) -> np.ndarray:
    """Standard causal (upper-triangle) mask."""
    return np.triu(np.ones((T, T), dtype=bool), k=1)


def make_local_mask(T: int, w: int = 2) -> np.ndarray:
    """
    Sliding window: token i attends to {max(0,i-w), …, i} only.
    Combined with causal masking.

    Forbidden positions: upper triangle (future) OR |i-j| > w.
    """
    causal   = np.triu(np.ones((T, T), dtype=bool), k=1)
    rows     = np.arange(T)[:, None]
    cols     = np.arange(T)[None, :]
    out_of_window = rows - cols > w       # tokens too far in the past
    return causal | out_of_window


def make_strided_mask(T: int, stride: int = 3) -> np.ndarray:
    """
    Strided: token i attends to
      - every token in [i-stride, i] (local recent window), AND
      - every stride-th past token: i-stride, i-2*stride, …

    Forbidden = upper triangle AND not local AND not strided.
    """
    causal  = np.triu(np.ones((T, T), dtype=bool), k=1)
    rows    = np.arange(T)[:, None]
    cols    = np.arange(T)[None, :]
    diff    = rows - cols      # ≥0 for past tokens (rows ≥ cols, causal side)

    # Local: diff in [0, stride]
    local   = (diff >= 0) & (diff <= stride)
    # Strided: diff > 0 and diff is a multiple of stride
    strided = (diff > 0) & (diff % stride == 0)

    allowed = local | strided
    # Forbidden: future (causal) OR not in allowed set
    return causal | (~causal & ~allowed)


def make_bigbird_mask(
    T: int,
    w: int = 2,
    n_random: int = 2,
    n_global: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    BigBird-style: random + local + global tokens.

    Each token attends to:
      - g global tokens (first n_global tokens attend to all; all attend to them)
      - its ±w local window
      - n_random random tokens (same random set for all queries, seeded)

    Combined with causal masking.
    """
    rng = np.random.default_rng(seed)
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)
    rows   = np.arange(T)[:, None]
    cols   = np.arange(T)[None, :]

    # Local window (causal side: attend to past ±w)
    local  = (rows - cols >= 0) & (rows - cols <= w)

    # Global: first n_global tokens can attend to everything; all attend to them
    global_col = np.zeros((T, T), dtype=bool)
    global_col[:, :n_global] = True       # all rows attend to first n_global cols
    global_row = np.zeros((T, T), dtype=bool)
    global_row[:n_global, :] = True       # first n_global rows attend to everything

    # Random: pick n_random columns for each row (causal — only past positions)
    rand_mask = np.zeros((T, T), dtype=bool)
    for i in range(T):
        past = np.arange(0, i + 1)        # causal: only past tokens
        if len(past) > n_random:
            chosen = rng.choice(past, size=n_random, replace=False)
            rand_mask[i, chosen] = True

    allowed = local | global_col | global_row | rand_mask
    return causal | (~causal & ~allowed)


# ─────────────────────────────────────────────────────────────────────────────
#  FLOPS estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_flops(mask: np.ndarray, d_head: int, n_heads: int) -> int:
    """
    Estimate attention FLOPS for one forward pass:
      QK^T : 2 * allowed_pairs * d_head
      AV   : 2 * allowed_pairs * d_head
    Multiply by n_heads.
    """
    T = mask.shape[0]
    allowed_pairs = int((~mask).sum())
    return 4 * allowed_pairs * d_head * n_heads


def sparsity_ratio(mask: np.ndarray) -> float:
    """Fraction of attention positions that are masked out (forbidden)."""
    T = mask.shape[0]
    # Exclude the trivially masked upper triangle (future tokens)
    causal_forbidden = T * (T - 1) // 2  # upper triangle without diagonal
    extra_forbidden  = int(mask.sum()) - causal_forbidden
    total_causal_allowed = T * (T + 1) // 2  # lower triangle + diagonal
    return extra_forbidden / total_causal_allowed


# ─────────────────────────────────────────────────────────────────────────────
#  Attention layer with a fixed boolean mask
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


def _cross_entropy_seq(logits, targets):
    N, T, V = logits.shape
    probs   = _softmax(logits.reshape(N * T, V))
    flat_t  = targets.reshape(N * T)
    loss    = -np.log(probs[np.arange(N * T), flat_t] + 1e-7).mean()
    return loss, probs.reshape(N, T, V)


class SparseAttention:
    """
    Multi-head self-attention with a precomputed boolean mask.

    mask[i, j] = True → position (i, j) is forbidden (set to -inf)
    """

    def __init__(self, d_model: int, n_heads: int, mask: np.ndarray):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self._mask   = mask          # (T, T) boolean

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

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, T, _ = x.shape
        Q = self._split(x @ self.Wq)
        K = self._split(x @ self.Wk)
        V = self._split(x @ self.Wv)

        scale  = 1.0 / np.sqrt(self.d_head)
        scores = Q @ K.transpose(0, 1, 3, 2) * scale   # (N, h, T, T)

        # Expand mask if needed (e.g., if sequence is shorter than prebuilt mask)
        T_mask = self._mask.shape[0]
        if T <= T_mask:
            m = self._mask[:T, :T]
        else:
            # Dynamically build full causal mask for longer sequences
            m = np.triu(np.ones((T, T), dtype=bool), k=1)

        scores[:, :, m] = -1e9

        A   = _softmax(scores)
        out = self._merge(A @ V) @ self.Wo

        self.cache = (x, Q, K, V, A)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, Q, K, V, A = self.cache
        N, T, _ = x.shape
        sc = 1.0 / np.sqrt(self.d_head)

        pre_Wo = self._merge(A @ V)
        self.grads[3][...] = pre_Wo.reshape(N * T, -1).T @ dout.reshape(N * T, -1)
        d_pre = dout @ self.Wo.T
        d_AV  = self._split(d_pre)

        dA = d_AV @ V.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ d_AV
        ds = A * (dA - (dA * A).sum(-1, keepdims=True)) * sc

        T_mask = self._mask.shape[0]
        m = self._mask[:T, :T] if T <= T_mask else np.triu(np.ones((T, T), bool), k=1)
        ds[:, :, m] = 0.0

        dQ = ds @ K
        dK = ds.transpose(0, 1, 3, 2) @ Q
        dQm = self._merge(dQ)
        dKm = self._merge(dK)
        dVm = self._merge(dV)

        self.grads[0][...] = x.reshape(N * T, -1).T @ dQm.reshape(N * T, -1)
        self.grads[1][...] = x.reshape(N * T, -1).T @ dKm.reshape(N * T, -1)
        self.grads[2][...] = x.reshape(N * T, -1).T @ dVm.reshape(N * T, -1)

        return (dQm @ self.Wq.T + dKm @ self.Wk.T + dVm @ self.Wv.T)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper layers
# ─────────────────────────────────────────────────────────────────────────────

class _LayerNorm:
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
        si = 1.0 / np.sqrt(var + self.eps)
        self.grads[0][...] = (dout * xh).sum(tuple(range(dout.ndim - 1)))
        self.grads[1][...] = dout.sum(tuple(range(dout.ndim - 1)))
        dxh = dout * self.g
        return si * (dxh - dxh.mean(-1, keepdims=True)
                     - xh * (dxh * xh).mean(-1, keepdims=True))


class _FFN:
    def __init__(self, d_model, d_ff):
        sc = np.sqrt(d_model)
        self.W1 = (np.random.randn(d_model, d_ff)    / sc).astype("f")
        self.b1 = np.zeros(d_ff,    dtype="f")
        self.W2 = (np.random.randn(d_ff,   d_model)  / np.sqrt(d_ff)).astype("f")
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
        xr, hr, dr = (x.reshape(-1, x.shape[-1]), h.reshape(-1, h.shape[-1]),
                      dout.reshape(-1, dout.shape[-1]))
        dW2, db2 = hr.T @ dr, dr.sum(0)
        dh = dr @ self.W2.T
        dh[hr == 0] = 0
        dW1, db1 = xr.T @ dh, dh.sum(0)
        self.grads[0][...] = dW1; self.grads[1][...] = db1
        self.grads[2][...] = dW2; self.grads[3][...] = db2
        return (dh @ self.W1.T).reshape(x.shape)


class SparseBlock:
    """Pre-LN block with pluggable SparseAttention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, mask: np.ndarray):
        self.ln1  = _LayerNorm(d_model)
        self.attn = SparseAttention(d_model, n_heads, mask)
        self.ln2  = _LayerNorm(d_model)
        self.ffn  = _FFN(d_model, d_ff)
        self.params = (self.ln1.params + self.attn.params
                       + self.ln2.params + self.ffn.params)
        self.grads  = (self.ln1.grads  + self.attn.grads
                       + self.ln2.grads  + self.ffn.grads)
        self.cache  = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        h   = x + self.attn.forward(self.ln1.forward(x))
        out = h + self.ffn.forward(self.ln2.forward(h))
        self.cache = (x, h)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, h = self.cache
        dh_f = self.ffn.backward(self.ln2.backward(dout))
        dh   = dout + dh_f
        dx_a = self.attn.backward(self.ln1.backward(dh))
        return dh + dx_a


# ─────────────────────────────────────────────────────────────────────────────
#  Positional encoding + tiny LM
# ─────────────────────────────────────────────────────────────────────────────

def _pe(T, d):
    pe  = np.zeros((T, d), dtype="f")
    pos = np.arange(T)[:, None]
    div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d // 2])
    return pe


class _SparseLM:
    def __init__(self, vocab_size, d_model, block, max_len=256):
        self.d_model    = d_model
        self.vocab_size = vocab_size
        sc = np.sqrt(d_model)
        self.embed_W = (np.random.randn(vocab_size, d_model) / sc).astype("f")
        self.head_b  = np.zeros(vocab_size, dtype="f")
        self.pe      = _pe(max_len, d_model)
        self.block   = block
        self.params  = [self.embed_W, self.head_b] + list(block.params)
        self.grads   = [np.zeros_like(self.embed_W), np.zeros_like(self.head_b)] + list(block.grads)
        self.cache   = None

    def forward(self, xs, ts):
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]
        self.emb_in = xs
        x = self.block.forward(x)
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
        dx  = (dl2 @ self.embed_W).reshape(N, T, self.d_model)
        dx  = self.block.backward(dx)
        dW_emb = np.zeros_like(self.embed_W)
        np.add.at(dW_emb, self.emb_in.reshape(-1), dx.reshape(N * T, self.d_model))
        self.grads[0][...] = dW_emb + dW_head


# ─────────────────────────────────────────────────────────────────────────────
#  Training runner
# ─────────────────────────────────────────────────────────────────────────────

def _train(model, corpus, epochs=200, batch_size=4, time_size=8, lr=1e-3):
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
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # Hyper-parameters for experiments
    D_MODEL    = 32
    N_HEADS    = 4
    D_FF       = 64
    EPOCHS     = 200
    BATCH_SIZE = 4
    TIME_SIZE  = 8
    T          = TIME_SIZE    # sequence length for mask generation
    W          = 2            # local window half-width
    STRIDE     = 3            # strided attention stride
    N_GLOBAL   = 1
    N_RANDOM   = 2
    SEED       = 42

    print("=" * 65)
    print("Sparse Attention Patterns — comparison experiment")
    print("=" * 65)

    # ── Build masks ──────────────────────────────────────────────────────────
    masks = {
        "Full":    make_full_mask(T),
        "Local":   make_local_mask(T, w=W),
        "Strided": make_strided_mask(T, stride=STRIDE),
        "BigBird": make_bigbird_mask(T, w=W, n_random=N_RANDOM, n_global=N_GLOBAL, seed=SEED),
    }

    print("\nAttention masks (T=8, rows=query, cols=key; True=blocked):")
    print(f"  {'':10s}  {'Full':8s}  {'Local(w=2)':12s}  {'Strided(s=3)':14s}  {'BigBird':10s}")
    for i in range(T):
        row = f"  token {i:2d}:  "
        for name, mask in masks.items():
            allowed = (~mask[i]).sum()
            row += f"attend {allowed}/{T}  "
        print(row)

    # ── FLOPS comparison ────────────────────────────────────────────────────
    print("\nEstimated FLOPS and sparsity per pattern:")
    flops_dict = {}
    for name, mask in masks.items():
        flops = estimate_flops(mask, D_MODEL // N_HEADS, N_HEADS)
        sp    = sparsity_ratio(mask)
        flops_dict[name] = flops
        print(f"  {name:10s}  FLOPS={flops:8,}  extra sparsity={sp*100:.1f}%")

    # ── Train models ─────────────────────────────────────────────────────────
    print(f"\nTraining {len(masks)} models ({EPOCHS} epochs each) …")
    ppl_hists = {}
    for name, mask in masks.items():
        np.random.seed(SEED)
        block = SparseBlock(D_MODEL, N_HEADS, D_FF, mask)
        model = _SparseLM(VOCAB_SIZE, D_MODEL, block)
        ppl_h = _train(model, CORPUS, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, time_size=TIME_SIZE)
        ppl_hists[name] = ppl_h
        print(f"  {name:10s}  final PPL={ppl_h[-1]:.2f}")

    # ── Compute Pareto (PPL vs FLOPS) ─────────────────────────────────────────
    final_ppls = {name: hist[-1] for name, hist in ppl_hists.items()}

    # ── Visualise ────────────────────────────────────────────────────────────
    colors = {
        "Full":    "steelblue",
        "Local":   "seagreen",
        "Strided": "salmon",
        "BigBird": "orange",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 1. Mask visualisations (2×2 subplot grids as subimages in axes[0,0])
    ax0 = axes[0, 0]
    n_pat = len(masks)
    img = np.zeros((T * 2 + 1, T * n_pat + (n_pat - 1)), dtype=float)
    for pi, (name, mask) in enumerate(masks.items()):
        x0 = pi * (T + 1)
        img[0:T, x0:x0 + T] = mask.astype(float)
    ax0.imshow(
        np.block([[m.astype(float) for m in masks.values()]]),
        cmap="RdBu_r", vmin=0, vmax=1, aspect="auto",
    )
    ax0.set_title("Attention Masks (blue=allowed, red=blocked)")
    ax0.set_xlabel("Key position")
    ax0.set_ylabel("Query position")
    # Label each mask section
    tick_xs = [T // 2 + i * T for i in range(n_pat)]
    ax0.set_xticks(tick_xs)
    ax0.set_xticklabels(list(masks.keys()), fontsize=8)

    # 2. PPL curves
    ep = np.arange(1, EPOCHS + 1)
    for name, hist in ppl_hists.items():
        axes[0, 1].plot(ep, hist, label=name, color=colors[name], linewidth=1.5)
    axes[0, 1].set_title("Perplexity vs Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Perplexity (lower = better)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Final PPL bar chart
    names3 = list(final_ppls.keys())
    ppls3  = [final_ppls[n] for n in names3]
    bars3  = axes[0, 2].bar(names3, ppls3,
                              color=[colors[n] for n in names3], edgecolor="k")
    for bar, ppl in zip(bars3, ppls3):
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{ppl:.1f}", ha="center", va="bottom", fontsize=9)
    axes[0, 2].set_title("Final Perplexity by Pattern")
    axes[0, 2].set_ylabel("Perplexity")
    axes[0, 2].grid(True, axis="y", alpha=0.3)

    # 4. FLOPS comparison bar chart
    names4 = list(flops_dict.keys())
    flops4 = [flops_dict[n] for n in names4]
    bars4  = axes[1, 0].bar(names4, flops4,
                              color=[colors[n] for n in names4], edgecolor="k")
    for bar, fl in zip(bars4, flops4):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{fl:,}", ha="center", va="bottom", fontsize=7)
    axes[1, 0].set_title("Estimated FLOPS per Forward Pass")
    axes[1, 0].set_ylabel("FLOPS")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    # 5. Pareto: PPL vs FLOPS scatter
    for name in masks:
        ppl_  = final_ppls[name]
        flops_ = flops_dict[name]
        axes[1, 1].scatter(flops_, ppl_, color=colors[name], s=80, zorder=5, label=name)
        axes[1, 1].annotate(name, (flops_, ppl_),
                             textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1, 1].set_title("PPL vs FLOPS (Pareto)")
    axes[1, 1].set_xlabel("Estimated FLOPS")
    axes[1, 1].set_ylabel("Final Perplexity")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Sparsity ratios
    sp_ratios = {name: sparsity_ratio(mask) * 100 for name, mask in masks.items()}
    names6 = list(sp_ratios.keys())
    vals6  = [sp_ratios[n] for n in names6]
    bars6  = axes[1, 2].bar(names6, vals6,
                              color=[colors[n] for n in names6], edgecolor="k")
    for bar, v in zip(bars6, vals6):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1, 2].set_ylim(0, max(vals6) * 1.3)
    axes[1, 2].set_title("Extra Sparsity vs Full-Causal Baseline")
    axes[1, 2].set_ylabel("Extra forbidden positions (%)")
    axes[1, 2].grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "Sparse Attention Patterns: Full / Local / Strided / BigBird",
        fontsize=13,
    )
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sparse_attention_comparison.png")
    plt.savefig(out_png, dpi=120)
    print(f"\nPlot saved to {out_png}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Pattern':10s}  {'PPL':8s}  {'FLOPS':10s}  {'Sparsity':10s}")
    print("-" * 65)
    for name in masks:
        print(f"{name:10s}  {final_ppls[name]:8.2f}  "
              f"{flops_dict[name]:10,}  {sp_ratios[name]:8.1f}%")
    print("=" * 65)
    print("\nDone.")
