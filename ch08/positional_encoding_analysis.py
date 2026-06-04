"""
ch08/positional_encoding_analysis.py — Positional Encoding Analysis

Analyses fixed sinusoidal PE vs learnable PE via:
  1. Heatmap of the PE matrix (positions × dimensions)
  2. Dot-product similarity between positions (should decrease with distance)
  3. Ablation: train with/without PE and compare final perplexity
  4. Comparison: sinusoidal vs learnable PE performance

Saves: ch08/pe_analysis.png

Usage
-----
    cd ch08
    python positional_encoding_analysis.py
"""

import sys
import os
sys.path.append("..")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.util import preprocess, clip_grads
from common.optimizer import Adam


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

def sinusoidal_pe(T, d_model):
    """Fixed sinusoidal positional encoding. Shape: (T, d_model)."""
    pe = np.zeros((T, d_model), dtype="f")
    pos = np.arange(T)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe


def random_learnable_pe(T, d_model, seed=0):
    """
    Simulated learnable PE (random initialization, not trained).
    In a full system this would be updated by backprop; here we compare
    the initial distribution to illustrate the structural difference.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, d_model)) * 0.02).astype("f")


# ---------------------------------------------------------------------------
# Minimal Transformer blocks (self-contained, no external imports beyond numpy)
# ---------------------------------------------------------------------------

def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy_seq(logits, targets):
    N, T, V = logits.shape
    probs = _softmax(logits.reshape(N * T, V))
    flat_t = targets.reshape(N * T)
    loss = -np.log(probs[np.arange(N * T), flat_t] + 1e-7).mean()
    return loss, probs.reshape(N, T, V)


class _LayerNorm:
    def __init__(self, d, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d, dtype="f")
        self.beta = np.zeros(d, dtype="f")
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]
        self.cache = None

    def forward(self, x):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        xh = (x - mu) / np.sqrt(var + self.eps)
        out = self.gamma * xh + self.beta
        self.cache = (x, xh, mu, var)
        return out

    def backward(self, dout):
        x, xh, mu, var = self.cache
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dgamma = (dout * xh).sum(axis=tuple(range(dout.ndim - 1)))
        dbeta = dout.sum(axis=tuple(range(dout.ndim - 1)))
        dxh = dout * self.gamma
        dx = std_inv * (dxh
                        - dxh.mean(-1, keepdims=True)
                        - xh * (dxh * xh).mean(-1, keepdims=True))
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta
        return dx


class _Attn:
    def __init__(self, d, h):
        self.d, self.h, self.dh = d, h, d // h
        s = np.sqrt(d)
        self.Wq = (np.random.randn(d, d) / s).astype("f")
        self.Wk = (np.random.randn(d, d) / s).astype("f")
        self.Wv = (np.random.randn(d, d) / s).astype("f")
        self.Wo = (np.random.randn(d, d) / s).astype("f")
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.grads = [np.zeros_like(p) for p in self.params]
        self.cache = None

    def _spl(self, x):
        N, T, _ = x.shape
        return x.reshape(N, T, self.h, self.dh).transpose(0, 2, 1, 3)

    def _mrg(self, x):
        N, h, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, T, h * dh)

    def forward(self, x):
        N, T, _ = x.shape
        Q = self._spl(x @ self.Wq)
        K = self._spl(x @ self.Wk)
        V = self._spl(x @ self.Wv)
        sc = np.sqrt(self.dh)
        scores = Q @ K.transpose(0, 1, 3, 2) / sc
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, :, mask] = -1e9
        A = _softmax(scores)
        out = self._mrg(A @ V) @ self.Wo
        self.cache = (x, Q, K, V, A)
        return out

    def backward(self, dout):
        x, Q, K, V, A, = self.cache
        N, T, _ = x.shape
        sc = np.sqrt(self.dh)
        pre = self._mrg(A @ V)
        dWo = pre.reshape(N * T, self.d).T @ dout.reshape(N * T, self.d)
        dm = dout @ self.Wo.T
        dAV = self._spl(dm)
        dA = dAV @ V.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ dAV
        ds = A * (dA - (dA * A).sum(-1, keepdims=True)) / sc
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        ds[:, :, mask] = 0
        dQ = ds @ K
        dK = ds.transpose(0, 1, 3, 2) @ Q
        dQm = self._mrg(dQ); dKm = self._mrg(dK); dVm = self._mrg(dV)
        dWq = x.reshape(N * T, self.d).T @ dQm.reshape(N * T, self.d)
        dWk = x.reshape(N * T, self.d).T @ dKm.reshape(N * T, self.d)
        dWv = x.reshape(N * T, self.d).T @ dVm.reshape(N * T, self.d)
        dx = dQm @ self.Wq.T + dKm @ self.Wk.T + dVm @ self.Wv.T
        for i, dw in enumerate([dWq, dWk, dWv, dWo]):
            self.grads[i][...] = dw
        return dx


class _FFN:
    def __init__(self, d, df):
        s = np.sqrt(d)
        self.W1 = (np.random.randn(d, df) / s).astype("f")
        self.b1 = np.zeros(df, dtype="f")
        self.W2 = (np.random.randn(df, d) / np.sqrt(df)).astype("f")
        self.b2 = np.zeros(d, dtype="f")
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.grads = [np.zeros_like(p) for p in self.params]
        self.cache = None

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        dout_r = dout.reshape(-1, dout.shape[-1])
        hr = h.reshape(-1, h.shape[-1])
        xr = x.reshape(-1, x.shape[-1])
        dW2 = hr.T @ dout_r; db2 = dout_r.sum(0)
        dh = dout_r @ self.W2.T; dh[hr == 0] = 0
        dW1 = xr.T @ dh; db1 = dh.sum(0)
        for i, dw in enumerate([dW1, db1, dW2, db2]):
            self.grads[i][...] = dw
        return (dh @ self.W1.T).reshape(x.shape)


class _Block:
    def __init__(self, d, h, df):
        self.n1 = _LayerNorm(d)
        self.attn = _Attn(d, h)
        self.n2 = _LayerNorm(d)
        self.ffn = _FFN(d, df)
        self.params = self.n1.params + self.attn.params + self.n2.params + self.ffn.params
        self.grads = self.n1.grads + self.attn.grads + self.n2.grads + self.ffn.grads
        self.cache = None

    def forward(self, x):
        h = x + self.attn.forward(self.n1.forward(x))
        out = h + self.ffn.forward(self.n2.forward(h))
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        dh = dout + self.ffn.backward(self.n2.backward(dout))
        return dh + self.attn.backward(self.n1.backward(dh))


class TinyTransformerLM:
    """
    Minimal Transformer LM that accepts a pre-computed PE matrix.
    pass pe=None to disable positional encoding entirely (ablation).
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 pe_matrix=None, max_len=256):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pe_matrix = pe_matrix  # (max_len, d_model) or None

        self.embed_W = (np.random.randn(vocab_size, d_model) / np.sqrt(d_model)).astype("f")
        self.blocks = [_Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.head_b = np.zeros(vocab_size, dtype="f")

        self.params = [self.embed_W, self.head_b]
        self.grads = [np.zeros_like(self.embed_W), np.zeros_like(self.head_b)]
        for blk in self.blocks:
            self.params += blk.params
            self.grads += blk.grads
        self.cache = None

    def forward(self, xs, ts):
        N, T = xs.shape
        x = self.embed_W[xs]
        if self.pe_matrix is not None:
            x = x + self.pe_matrix[:T]
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
        dlogits = probs.copy()
        dlogits[np.arange(N)[:, None], np.arange(T)[None, :], ts] -= 1
        dlogits *= dout / (N * T)
        self.grads[1][...] = dlogits.sum(axis=(0, 1))
        dlogits_2d = dlogits.reshape(N * T, V)
        x_2d = x.reshape(N * T, self.d_model)
        dW_head = dlogits_2d.T @ x_2d
        dx = (dlogits_2d @ self.embed_W).reshape(N, T, self.d_model)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        dW_emb = np.zeros_like(self.embed_W)
        np.add.at(dW_emb, self.emb_in.reshape(-1), dx.reshape(N * T, self.d_model))
        self.grads[0][...] = dW_emb + dW_head


def run_training(corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
                 pe_matrix, batch_size, time_size, max_epoch, label):
    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    max_iter = max(1, data_size // (batch_size * time_size))

    np.random.seed(42)  # reproducible init
    model = TinyTransformerLM(vocab_size, d_model, n_heads, n_layers, d_ff,
                               pe_matrix=pe_matrix)
    optimizer = Adam(lr=1e-3)
    ppl_history = []
    time_idx = 0

    for epoch in range(max_epoch):
        total_loss = total_count = 0
        for _ in range(max_iter):
            batch_xs = np.zeros((batch_size, time_size), dtype=np.int32)
            batch_ts = np.zeros((batch_size, time_size), dtype=np.int32)
            offsets = [data_size * i // batch_size for i in range(batch_size)]
            for t in range(time_size):
                for b in range(batch_size):
                    batch_xs[b, t] = xs[(offsets[b] + time_idx) % data_size]
                    batch_ts[b, t] = ts[(offsets[b] + time_idx) % data_size]
            time_idx = (time_idx + time_size) % data_size
            loss = model.forward(batch_xs, batch_ts)
            model.backward()
            clip_grads(model.grads, 1.0)
            optimizer.update(model.params, model.grads)
            total_loss += loss
            total_count += 1

        ppl = float(np.exp(total_loss / total_count))
        ppl_history.append(ppl)
        if epoch % 50 == 0 or epoch == max_epoch - 1:
            print(f"  [{label}]  epoch {epoch + 1:>3}  ppl={ppl:.2f}")

    return ppl_history


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pe_heatmap(ax, pe_matrix, title, max_pos=50, max_dim=64):
    """Draw a heatmap of the PE matrix."""
    mat = pe_matrix[:max_pos, :max_dim]
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("Dimension index")
    ax.set_ylabel("Position")
    ax.set_title(title)
    return im


def plot_position_similarity(ax, pe_matrix, n_pos=20):
    """
    Plot dot-product similarity between position 0 and all other positions.
    Expected: decreases with distance for sinusoidal PE.
    """
    norms = np.linalg.norm(pe_matrix, axis=1, keepdims=True) + 1e-8
    pe_norm = pe_matrix / norms
    sims = pe_norm[:n_pos] @ pe_norm[0]   # (n_pos,)
    ax.plot(range(n_pos), sims, marker="o", markersize=4)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Position j")
    ax.set_ylabel("cos-sim(PE[0], PE[j])")
    ax.set_title("Positional Similarity to Position 0")


def plot_ppl_comparison(ax, histories, labels, colors):
    """Plot perplexity curves for multiple runs."""
    for hist, label, color in zip(histories, labels, colors):
        ax.plot(range(1, len(hist) + 1), hist, label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Training Perplexity: PE Ablation")
    ax.legend()
    ax.set_yscale("log")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    text = ("the dog ran . the cat sat . the dog sat . "
            "a cat ran . a dog ran . the cat ran . "
            "the dog ate . a cat ate . the cat ate .")
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)

    d_model = 32
    n_heads = 4
    n_layers = 2
    d_ff = 64
    max_len = 64
    batch_size = 4
    time_size = 5
    max_epoch = 200

    # ---- Generate PE matrices ----
    sin_pe = sinusoidal_pe(max_len, d_model)
    lrn_pe = random_learnable_pe(max_len, d_model, seed=7)

    print("Sinusoidal PE shape:", sin_pe.shape)
    print("Learnable PE shape :", lrn_pe.shape)

    # ---- Training ablations ----
    print("\n--- Ablation: No PE ---")
    hist_no_pe = run_training(
        corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
        pe_matrix=None,
        batch_size=batch_size, time_size=time_size, max_epoch=max_epoch,
        label="No PE"
    )

    print("\n--- Ablation: Sinusoidal PE ---")
    hist_sin = run_training(
        corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
        pe_matrix=sin_pe,
        batch_size=batch_size, time_size=time_size, max_epoch=max_epoch,
        label="Sinusoidal PE"
    )

    print("\n--- Ablation: Learnable (random init) PE ---")
    hist_lrn = run_training(
        corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
        pe_matrix=lrn_pe,
        batch_size=batch_size, time_size=time_size, max_epoch=max_epoch,
        label="Learnable PE"
    )

    # ---- Report ----
    print("\n=== Final Perplexity Comparison ===")
    print(f"  No PE          : {hist_no_pe[-1]:.2f}")
    print(f"  Sinusoidal PE  : {hist_sin[-1]:.2f}")
    print(f"  Learnable PE   : {hist_lrn[-1]:.2f}")

    # ---- Figures ----
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Positional Encoding Analysis", fontsize=15)

    # Row 1 – PE heatmaps
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax3 = fig.add_subplot(3, 3, 3)

    im1 = plot_pe_heatmap(ax1, sin_pe, "Sinusoidal PE (positions × dims)")
    im2 = plot_pe_heatmap(ax2, lrn_pe, "Learnable PE – random init")

    # Difference map
    diff = sin_pe[:50, :64] - lrn_pe[:50, :64]
    im3 = ax3.imshow(diff, aspect="auto", cmap="PRGn",
                     vmin=-diff.std() * 2, vmax=diff.std() * 2)
    ax3.set_title("Sinusoidal − Learnable (difference)")
    ax3.set_xlabel("Dimension"); ax3.set_ylabel("Position")

    for im, ax in [(im1, ax1), (im2, ax2), (im3, ax3)]:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2 – Position similarity
    ax4 = fig.add_subplot(3, 2, 3)
    ax5 = fig.add_subplot(3, 2, 4)
    plot_position_similarity(ax4, sin_pe, n_pos=min(30, max_len))
    ax4.set_title("Sinusoidal PE: Position Similarity")

    plot_position_similarity(ax5, lrn_pe, n_pos=min(30, max_len))
    ax5.set_title("Learnable PE: Position Similarity")

    # Row 3 – PPL comparison
    ax6 = fig.add_subplot(3, 1, 3)
    plot_ppl_comparison(
        ax6,
        histories=[hist_no_pe, hist_sin, hist_lrn],
        labels=["No PE", "Sinusoidal PE", "Learnable PE"],
        colors=["#e74c3c", "#2980b9", "#27ae60"],
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "pe_analysis.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {save_path}")
    print("Done.")
