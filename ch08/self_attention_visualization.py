"""
ch08/self_attention_visualization.py — Attention Weight Visualization

Train a small Transformer LM on the toy corpus, then extract attention weights
from each layer and head.  Produces two figures:
  • ch08/attention_visualization.png  — grid of heatmaps (layers × heads)
  • Printed report: which heads attend to recent tokens vs long-range

Usage
-----
    cd ch08
    python self_attention_visualization.py
"""

import sys
import os
sys.path.append("..")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from common.util import preprocess, clip_grads
from common.optimizer import Adam


# ---------------------------------------------------------------------------
# Reuse transformer building blocks from transformer_lm.py
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


class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model, dtype="f")
        self.beta = np.zeros(d_model, dtype="f")
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]
        self.cache = None

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        out = self.gamma * xhat + self.beta
        self.cache = (x, xhat, mu, var)
        return out

    def backward(self, dout):
        x, xhat, mu, var = self.cache
        N = x.shape[-1]
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dgamma = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
        dbeta = dout.sum(axis=tuple(range(dout.ndim - 1)))
        dxhat = dout * self.gamma
        dx = std_inv * (dxhat
                        - dxhat.mean(axis=-1, keepdims=True)
                        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta
        return dx


class CausalSelfAttentionViz:
    """
    Same as CausalSelfAttention but also caches the raw attention maps
    so they can be extracted after a forward pass without re-running.
    """

    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        scale = np.sqrt(d_model)
        self.Wq = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.Wk = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.Wv = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.Wo = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.grads = [np.zeros_like(p) for p in self.params]
        self.cache = None
        # For visualization: attention weights (n_heads, T, T)
        self.last_attn_weights = None

    def _split_heads(self, x):
        N, T, _ = x.shape
        return x.reshape(N, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        N, h, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, T, h * dh)

    def forward(self, x):
        N, T, _ = x.shape
        Q = self._split_heads(x @ self.Wq)
        K = self._split_heads(x @ self.Wk)
        V = self._split_heads(x @ self.Wv)

        scale = np.sqrt(self.d_head)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale

        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, :, mask] = -1e9

        A = _softmax(scores)
        # Cache first batch item's attention weights: (n_heads, T, T)
        self.last_attn_weights = A[0].copy()

        out = self._merge_heads(A @ V)
        out = out @ self.Wo
        self.cache = (x, Q, K, V, A, scores)
        return out

    def backward(self, dout):
        x, Q, K, V, A, scores = self.cache
        N, T, _ = x.shape
        scale = np.sqrt(self.d_head)

        out_pre_Wo = self._merge_heads(A @ V)
        dWo = out_pre_Wo.reshape(N * T, self.d_model).T @ dout.reshape(N * T, self.d_model)
        d_merged = dout @ self.Wo.T
        d_AV = self._split_heads(d_merged)

        dA = d_AV @ V.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ d_AV

        dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dscores /= scale
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        dscores[:, :, mask] = 0.0

        dQ = dscores @ K
        dK = dscores.transpose(0, 1, 3, 2) @ Q

        dQ_m = self._merge_heads(dQ)
        dK_m = self._merge_heads(dK)
        dV_m = self._merge_heads(dV)

        dWq = x.reshape(N * T, self.d_model).T @ dQ_m.reshape(N * T, self.d_model)
        dWk = x.reshape(N * T, self.d_model).T @ dK_m.reshape(N * T, self.d_model)
        dWv = x.reshape(N * T, self.d_model).T @ dV_m.reshape(N * T, self.d_model)

        dx = dQ_m @ self.Wq.T + dK_m @ self.Wk.T + dV_m @ self.Wv.T
        self.grads[0][...] = dWq
        self.grads[1][...] = dWk
        self.grads[2][...] = dWv
        self.grads[3][...] = dWo
        return dx


class FFN:
    def __init__(self, d_model, d_ff):
        scale = np.sqrt(d_model)
        self.W1 = (np.random.randn(d_model, d_ff) / scale).astype("f")
        self.b1 = np.zeros(d_ff, dtype="f")
        self.W2 = (np.random.randn(d_ff, d_model) / np.sqrt(d_ff)).astype("f")
        self.b2 = np.zeros(d_model, dtype="f")
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
        xr = x.reshape(-1, x.shape[-1])
        hr = h.reshape(-1, h.shape[-1])
        dout_r = dout.reshape(-1, dout.shape[-1])
        dW2 = hr.T @ dout_r
        db2 = dout_r.sum(axis=0)
        dh = dout_r @ self.W2.T
        dh[hr == 0] = 0
        dW1 = xr.T @ dh
        db1 = dh.sum(axis=0)
        dx = (dh @ self.W1.T).reshape(x.shape)
        self.grads[0][...] = dW1
        self.grads[1][...] = db1
        self.grads[2][...] = dW2
        self.grads[3][...] = db2
        return dx


class TransformerBlockViz:
    def __init__(self, d_model, n_heads, d_ff):
        self.norm1 = LayerNorm(d_model)
        self.attn = CausalSelfAttentionViz(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.params = (self.norm1.params + self.attn.params
                       + self.norm2.params + self.ffn.params)
        self.grads = (self.norm1.grads + self.attn.grads
                      + self.norm2.grads + self.ffn.grads)
        self.cache = None

    def forward(self, x):
        h = x + self.attn.forward(self.norm1.forward(x))
        out = h + self.ffn.forward(self.norm2.forward(h))
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        dh_ffn = self.ffn.backward(self.norm2.backward(dout))
        dh = dout + dh_ffn
        dx_attn = self.attn.backward(self.norm1.backward(dh))
        return dh + dx_attn


def positional_encoding(T, d_model):
    pe = np.zeros((T, d_model), dtype="f")
    pos = np.arange(T)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe


class TransformerLMViz:
    """TransformerLM with per-layer attention weight extraction."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=256):
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed_W = (np.random.randn(vocab_size, d_model) / np.sqrt(d_model)).astype("f")
        self.pe = positional_encoding(max_len, d_model)

        self.blocks = [TransformerBlockViz(d_model, n_heads, d_ff)
                       for _ in range(n_layers)]
        self.head_b = np.zeros(vocab_size, dtype="f")

        self.params = [self.embed_W, self.head_b]
        self.grads = [np.zeros_like(self.embed_W), np.zeros_like(self.head_b)]
        for blk in self.blocks:
            self.params += blk.params
            self.grads += blk.grads

        self.cache = None

    def forward(self, xs, ts):
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]
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

    def get_attention_weights(self, token_ids):
        """
        Run a single forward pass on token_ids and return attention maps.

        Parameters
        ----------
        token_ids : 1-D array of int
            Token IDs for the sample sentence.

        Returns
        -------
        list of ndarray, shape (n_heads, T, T)
            One entry per transformer layer.
        """
        xs = np.array(token_ids, dtype=np.int32)[np.newaxis]
        T = xs.shape[1]
        x = self.embed_W[xs] + self.pe[:T]
        attn_maps = []
        for blk in self.blocks:
            x = blk.forward(x)
            attn_maps.append(blk.attn.last_attn_weights.copy())
        return attn_maps

    def generate(self, start_id, length, max_len=256):
        tokens = [start_id]
        for _ in range(length - 1):
            xs = np.array(tokens, dtype=np.int32)[np.newaxis]
            T = xs.shape[1]
            x = self.embed_W[xs] + self.pe[:T]
            for blk in self.blocks:
                x = blk.forward(x)
            logits = x[0, -1] @ self.embed_W.T + self.head_b
            next_id = int(logits.argmax())
            tokens.append(next_id)
        return tokens


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
          batch_size, time_size, max_epoch):
    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    max_iter = max(1, data_size // (batch_size * time_size))

    model = TransformerLMViz(vocab_size, d_model, n_heads, n_layers, d_ff)
    optimizer = Adam(lr=1e-3)

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
        if epoch % 50 == 0 or epoch == max_epoch - 1:
            print(f"  epoch {epoch + 1:>3}  ppl={ppl:.2f}")

    return model


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def attention_entropy(attn_map):
    """
    Compute per-head average entropy of the attention distribution.

    attn_map : (n_heads, T, T)  — rows are distributions over T positions
    Returns   : (n_heads,)       entropy values
    """
    eps = 1e-9
    # Only use lower-triangular part (causal), skip 0-length rows
    T = attn_map.shape[-1]
    entropies = []
    for h in range(attn_map.shape[0]):
        head_ent = []
        for t in range(T):
            row = attn_map[h, t, :t + 1]  # valid tokens up to position t
            row = np.maximum(row, eps)
            row = row / row.sum()
            ent = -np.sum(row * np.log(row))
            head_ent.append(ent)
        entropies.append(np.mean(head_ent))
    return np.array(entropies)


def mean_attended_distance(attn_map):
    """
    Average token distance the attention focuses on.
    High value → long-range; Low value → local/recent.
    attn_map : (n_heads, T, T)
    """
    T = attn_map.shape[-1]
    distances = []
    for h in range(attn_map.shape[0]):
        head_dists = []
        for t in range(1, T):
            row = attn_map[h, t, :t + 1]
            positions = np.arange(t + 1, dtype=float)
            dist = np.sum(row * (t - positions))
            head_dists.append(dist)
        distances.append(np.mean(head_dists) if head_dists else 0.0)
    return np.array(distances)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attention_visualization(attn_maps, tokens, save_path):
    """
    attn_maps : list of (n_heads, T, T), one per layer
    tokens    : list of str
    """
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[0]
    T = len(tokens)

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(3 * n_heads, 3 * n_layers),
        squeeze=False,
    )
    fig.suptitle("Attention Weights per Layer and Head", fontsize=14, y=1.02)

    for layer_idx, attn in enumerate(attn_maps):
        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]
            mat = attn[head_idx]  # (T, T)
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set_title(f"L{layer_idx + 1} H{head_idx + 1}", fontsize=9)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(tokens, fontsize=7)
            if layer_idx == n_layers - 1:
                ax.set_xlabel("Key position", fontsize=8)
            if head_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx + 1}\nQuery", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved attention heatmaps → {save_path}")


def plot_entropy_and_distance(attn_maps, save_path):
    """
    Two sub-plots:
      1. Attention entropy per head per layer (bar chart)
      2. Mean attended distance per head per layer (bar chart)
    """
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[0]
    head_labels = [f"H{h + 1}" for h in range(n_heads)]

    entropies = np.array([attention_entropy(a) for a in attn_maps])   # (L, H)
    distances = np.array([mean_attended_distance(a) for a in attn_maps])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Entropy
    ax = axes[0]
    x = np.arange(n_heads)
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    for l in range(n_layers):
        ax.bar(x + l * 0.2, entropies[l], width=0.18,
               label=f"Layer {l + 1}", color=colors[l], alpha=0.85)
    ax.set_xticks(x + (n_layers - 1) * 0.1)
    ax.set_xticklabels(head_labels)
    ax.set_xlabel("Head")
    ax.set_ylabel("Average Entropy (nats)")
    ax.set_title("Attention Entropy per Head\n(lower = more focused)")
    ax.legend(fontsize=8)

    # Distance
    ax = axes[1]
    for l in range(n_layers):
        ax.bar(x + l * 0.2, distances[l], width=0.18,
               label=f"Layer {l + 1}", color=colors[l], alpha=0.85)
    ax.set_xticks(x + (n_layers - 1) * 0.1)
    ax.set_xticklabels(head_labels)
    ax.set_xlabel("Head")
    ax.set_ylabel("Mean Attended Distance (tokens)")
    ax.set_title("Mean Attended Token Distance per Head\n(higher = longer-range)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved entropy/distance plots → {save_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(attn_maps, head_threshold_recent=0.8):
    """
    Print a plain-text report describing which heads focus locally vs globally.
    """
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[0]

    entropies = np.array([attention_entropy(a) for a in attn_maps])
    distances = np.array([mean_attended_distance(a) for a in attn_maps])
    max_dist = distances.max() + 1e-9

    print("\n=== Attention Head Analysis ===")
    print(f"{'Layer':>6}  {'Head':>5}  {'Entropy':>9}  {'Dist':>8}  Pattern")
    print("-" * 55)
    for l in range(n_layers):
        for h in range(n_heads):
            ent = entropies[l, h]
            dist = distances[l, h]
            rel_dist = dist / max_dist
            if ent < 0.5 and rel_dist < 0.3:
                pattern = "FOCUSED  – attends to very recent tokens"
            elif ent < 0.5 and rel_dist >= 0.3:
                pattern = "FOCUSED  – attends to distant tokens (long-range)"
            elif ent >= 1.0:
                pattern = "DIFFUSE  – spreads attention broadly"
            else:
                pattern = "MIXED    – intermediate focus"
            print(f"  L{l + 1:>2}     H{h + 1:>2}   {ent:>8.3f}  {dist:>7.2f}  {pattern}")


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
    words = text.lower().replace(".", " .").split()

    d_model = 32
    n_heads = 4
    n_layers = 2
    d_ff = 64
    batch_size = 4
    time_size = 5
    max_epoch = 300

    print(f"Vocabulary: {word_to_id}")
    print(f"Corpus length: {len(corpus)}")
    print(f"\nTraining Transformer ({n_layers} layers, {n_heads} heads)…")

    model = train(corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
                  batch_size, time_size, max_epoch)

    # --- Select a representative sample sentence for visualization ---
    # Use "the dog ran . the cat sat" (first 6 tokens)
    sample_text = "the dog ran . the cat"
    sample_tokens = sample_text.lower().split()
    sample_ids = [word_to_id[w] for w in sample_tokens if w in word_to_id]

    print(f"\nSample sentence: {sample_tokens}")
    print(f"Token IDs      : {sample_ids}")

    attn_maps = model.get_attention_weights(sample_ids)

    # ---- Plots ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_path = os.path.join(script_dir, "attention_visualization.png")
    ent_path = os.path.join(script_dir, "attention_entropy.png")

    plot_attention_visualization(attn_maps, sample_tokens, viz_path)
    plot_entropy_and_distance(attn_maps, ent_path)

    report(attn_maps)

    # ---- Quick generation sanity check ----
    start_id = word_to_id.get("the", 0)
    gen_ids = model.generate(start_id, length=8)
    gen_words = [id_to_word.get(i, "?") for i in gen_ids]
    print(f"\nGenerated: {' '.join(gen_words)}")
    print("\nDone.")
