"""
Pure NumPy causal (decoder-only) Transformer language model.

Architecture
------------
  token embedding + positional encoding
  → N x TransformerBlock (LayerNorm, CausalSelfAttention, FFN)
  → linear head (weight-tied to embedding)
  → TimeSoftmaxWithLoss
"""

import sys
sys.path.append("..")
import numpy as np
from common.util import preprocess, clip_grads
from common.optimizer import Adam


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy_seq(logits, targets):
    """
    logits : (N, T, V)
    targets: (N, T)  integer class indices
    """
    N, T, V = logits.shape
    probs = _softmax(logits.reshape(N*T, V))
    flat_t = targets.reshape(N*T)
    loss = -np.log(probs[np.arange(N*T), flat_t] + 1e-7).mean()
    return loss, probs.reshape(N, T, V)


# ---------------------------------------------------------------------------
# Layer normalisation (no trainable scale/shift for simplicity)
# ---------------------------------------------------------------------------

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model, dtype="f")
        self.beta  = np.zeros(d_model, dtype="f")
        self.params = [self.gamma, self.beta]
        self.grads  = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]
        self.cache  = None

    def forward(self, x):
        mu  = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1,  keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        out  = self.gamma * xhat + self.beta
        self.cache = (x, xhat, mu, var)
        return out

    def backward(self, dout):
        x, xhat, mu, var = self.cache
        N = x.shape[-1]
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dgamma = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
        dbeta  = dout.sum(axis=tuple(range(dout.ndim - 1)))
        dxhat  = dout * self.gamma
        # gradient of LN w.r.t. x
        dx = std_inv * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta
        return dx


# ---------------------------------------------------------------------------
# Causal (masked) multi-head self-attention
# ---------------------------------------------------------------------------

class CausalSelfAttention:
    """
    Multi-head self-attention with upper-triangle causal mask.
    Weights: Wq, Wk, Wv  (d_model, d_model),  Wo (d_model, d_model)
    """

    def __init__(self, d_model, n_heads):
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
        self.cache  = None

    def _split_heads(self, x):
        # x: (N, T, d_model) -> (N, h, T, d_head)
        N, T, _ = x.shape
        x = x.reshape(N, T, self.n_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        # (N, h, T, d_head) -> (N, T, d_model)
        N, h, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, T, h * dh)

    def forward(self, x):
        N, T, _ = x.shape
        Q = self._split_heads(x @ self.Wq)
        K = self._split_heads(x @ self.Wk)
        V = self._split_heads(x @ self.Wv)

        scale = np.sqrt(self.d_head)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale   # (N, h, T, T)

        # causal mask: future positions set to -inf
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, :, mask] = -1e9

        A = _softmax(scores)    # (N, h, T, T)
        out = self._merge_heads(A @ V)    # (N, T, d_model)
        out = out @ self.Wo

        self.cache = (x, Q, K, V, A, scores)
        return out

    def backward(self, dout):
        x, Q, K, V, A, scores = self.cache
        N, T, _ = x.shape
        scale = np.sqrt(self.d_head)

        # dWo
        out_pre_Wo = self._merge_heads(A @ V)
        dWo = out_pre_Wo.reshape(N*T, self.d_model).T @ dout.reshape(N*T, self.d_model)
        d_merged = dout @ self.Wo.T

        # split back to heads
        d_AV = self._split_heads(d_merged)   # (N, h, T, dh)

        # dA and dV
        dA = d_AV @ V.transpose(0, 1, 3, 2)   # (N, h, T, T)
        dV = A.transpose(0, 1, 3, 2) @ d_AV   # (N, h, T, dh)

        # softmax backward  (element-wise): dscores = A * (dA - (dA*A).sum(-1, keepdims))
        dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dscores /= scale

        # causal positions were -inf → zero gradient there
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        dscores[:, :, mask] = 0.0

        dQ = dscores @ K              # (N, h, T, dh)
        dK = dscores.transpose(0, 1, 3, 2) @ Q  # (N, h, T, dh)

        # merge heads and compute input grads
        dQ_m = self._merge_heads(dQ)
        dK_m = self._merge_heads(dK)
        dV_m = self._merge_heads(dV)

        dWq = x.reshape(N*T, self.d_model).T @ dQ_m.reshape(N*T, self.d_model)
        dWk = x.reshape(N*T, self.d_model).T @ dK_m.reshape(N*T, self.d_model)
        dWv = x.reshape(N*T, self.d_model).T @ dV_m.reshape(N*T, self.d_model)

        dx = (dQ_m @ self.Wq.T + dK_m @ self.Wk.T + dV_m @ self.Wv.T)

        self.grads[0][...] = dWq
        self.grads[1][...] = dWk
        self.grads[2][...] = dWv
        self.grads[3][...] = dWo
        return dx


# ---------------------------------------------------------------------------
# Position-wise FFN
# ---------------------------------------------------------------------------

class FFN:
    def __init__(self, d_model, d_ff):
        scale = np.sqrt(d_model)
        self.W1 = (np.random.randn(d_model, d_ff)  / scale).astype("f")
        self.b1 = np.zeros(d_ff).astype("f")
        self.W2 = (np.random.randn(d_ff,   d_model) / np.sqrt(d_ff)).astype("f")
        self.b2 = np.zeros(d_model).astype("f")
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.grads  = [np.zeros_like(p) for p in self.params]
        self.cache  = None

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        out = h @ self.W2 + self.b2
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        N_T = x.shape[0] * x.shape[1] if x.ndim == 3 else x.shape[0]
        xr  = x.reshape(-1, x.shape[-1])
        hr  = h.reshape(-1, h.shape[-1])
        dout_r = dout.reshape(-1, dout.shape[-1])
        dW2 = hr.T @ dout_r
        db2 = dout_r.sum(axis=0)
        dh  = dout_r @ self.W2.T
        dh[hr == 0] = 0   # ReLU gate
        dW1 = xr.T @ dh
        db1 = dh.sum(axis=0)
        dx  = (dh @ self.W1.T).reshape(x.shape)
        self.grads[0][...] = dW1
        self.grads[1][...] = db1
        self.grads[2][...] = dW2
        self.grads[3][...] = db2
        return dx


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.norm1 = LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.ffn   = FFN(d_model, d_ff)
        self.params = (self.norm1.params + self.attn.params
                       + self.norm2.params + self.ffn.params)
        self.grads  = (self.norm1.grads  + self.attn.grads
                       + self.norm2.grads  + self.ffn.grads)
        self.cache = None

    def forward(self, x):
        h = x + self.attn.forward(self.norm1.forward(x))
        out = h + self.ffn.forward(self.norm2.forward(h))
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        # FFN residual
        dh_ffn = self.ffn.backward(self.norm2.backward(dout))
        dh = dout + dh_ffn
        # Attention residual
        dx_attn = self.attn.backward(self.norm1.backward(dh))
        dx = dh + dx_attn
        return dx


# ---------------------------------------------------------------------------
# Positional encoding (fixed sinusoidal)
# ---------------------------------------------------------------------------

def positional_encoding(T, d_model):
    pe = np.zeros((T, d_model), dtype="f")
    pos = np.arange(T)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe


# ---------------------------------------------------------------------------
# TransformerLM
# ---------------------------------------------------------------------------

class TransformerLM:
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=256):
        self.d_model    = d_model
        self.vocab_size = vocab_size

        self.embed_W = (np.random.randn(vocab_size, d_model) / np.sqrt(d_model)).astype("f")
        self.pe      = positional_encoding(max_len, d_model)

        self.blocks = [TransformerBlock(d_model, n_heads, d_ff)
                       for _ in range(n_layers)]

        # LM head weight-tied to embedding (head_W = embed_W.T)
        self.head_b = np.zeros(vocab_size, dtype="f")

        # Collect params / grads
        self.params = [self.embed_W, self.head_b]
        self.grads  = [np.zeros_like(self.embed_W), np.zeros_like(self.head_b)]
        for blk in self.blocks:
            self.params += blk.params
            self.grads  += blk.grads

        # Cache for backward
        self.cache = None

    # ------------------------------------------------------------------

    def forward(self, xs, ts):
        """
        xs: (N, T) integer token ids
        ts: (N, T) integer target ids (next-token)
        Returns scalar loss.
        """
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]   # (N, T, d_model)
        self.emb_in = xs

        for blk in self.blocks:
            x = blk.forward(x)

        # LM head (weight-tied)
        logits = x @ self.embed_W.T + self.head_b   # (N, T, V)
        loss, probs = _cross_entropy_seq(logits, ts)

        self.cache = (x, probs, ts)
        return loss

    def backward(self, dout=1):
        x, probs, ts = self.cache
        N, T, V = probs.shape

        # dlogits from cross-entropy
        dlogits = probs.copy()
        dlogits[np.arange(N)[:, None], np.arange(T)[None, :], ts] -= 1
        dlogits *= dout / (N * T)

        # LM head
        self.grads[1][...] = dlogits.sum(axis=(0, 1))     # db
        # grad w.r.t embed_W from head (dlogits @ embed_W is the matmul)
        dlogits_2d = dlogits.reshape(N*T, V)
        x_2d = x.reshape(N*T, self.d_model)
        dW_head = dlogits_2d.T @ x_2d                     # (V, d_model)
        dx = dlogits_2d @ self.embed_W                     # (N*T, d_model)
        dx = dx.reshape(N, T, self.d_model)

        # Transformer blocks
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)

        # grad w.r.t. embed_W from token embedding (input side)
        dW_emb = np.zeros_like(self.embed_W)
        np.add.at(dW_emb, self.emb_in.reshape(-1), dx.reshape(N*T, self.d_model))

        self.grads[0][...] = dW_emb + dW_head

    # ------------------------------------------------------------------

    def generate(self, start_id, length, max_len=256):
        """Greedy autoregressive generation."""
        tokens = [start_id]
        for _ in range(length - 1):
            xs = np.array(tokens, dtype=np.int32)[np.newaxis]   # (1, t)
            T = xs.shape[1]
            x = self.embed_W[xs] + self.pe[:T]
            for blk in self.blocks:
                x = blk.forward(x)
            logits = x[0, -1] @ self.embed_W.T + self.head_b    # (V,)
            next_id = int(logits.argmax())
            tokens.append(next_id)
        return tokens


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
                 batch_size, time_size, max_epoch, label):
    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    max_iter  = max(1, data_size // (batch_size * time_size))

    model     = TransformerLM(vocab_size, d_model, n_heads, n_layers, d_ff)
    optimizer = Adam(lr=1e-3)
    ppl_list  = []

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
            total_loss   += loss
            total_count  += 1

        ppl = float(np.exp(total_loss / total_count))
        ppl_list.append(ppl)
        if epoch % 50 == 0:
            print(f"[{label}] epoch {epoch+1:>3}  ppl={ppl:.2f}")

    return model, ppl_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    text = ("the dog ran . the cat sat . the dog sat . "
            "a cat ran . a dog ran . the cat ran . "
            "the dog ate . a cat ate . the cat ate .")
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)

    # Tiny Transformer settings suited for this small corpus
    d_model   = 32
    n_heads   = 4
    n_layers  = 2
    d_ff      = 64
    batch_size = 4
    time_size  = 5
    max_epoch  = 200

    transformer_model, tf_ppls = run_training(
        corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
        batch_size, time_size, max_epoch, label="Transformer"
    )

    # Compare against SimpleRNNLM
    sys.path.insert(0, "..")
    from ch05.rnnlm import SimpleRNNLM
    from ch06.lstm_lm import BetterRNNLM
    from common.optimizer import SGD

    def train_rnn(model, corpus, batch_size, time_size, max_epoch, lr, max_grad, label):
        xs, ts = corpus[:-1], corpus[1:]
        data_size = len(xs)
        max_iter  = max(1, data_size // (batch_size * time_size))
        optimizer = SGD(lr)
        time_idx  = 0
        ppl_list  = []
        for epoch in range(max_epoch):
            if hasattr(model, "reset_state"):
                model.reset_state()
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
                clip_grads(model.grads, max_grad)
                optimizer.update(model.params, model.grads)
                total_loss += loss
                total_count += 1
            ppl = float(np.exp(total_loss / total_count))
            ppl_list.append(ppl)
            if epoch % 50 == 0:
                print(f"[{label}] epoch {epoch+1:>3}  ppl={ppl:.2f}")
        return ppl_list

    rnn_model  = SimpleRNNLM(vocab_size, 32, 32)
    rnn_ppls   = train_rnn(rnn_model,  corpus, batch_size, time_size, max_epoch, 0.1, 0.25, "RNN")

    lstm_model = BetterRNNLM(vocab_size, 32, 32)
    lstm_ppls  = train_rnn(lstm_model, corpus, batch_size, time_size, max_epoch, 20.0, 0.25, "LSTM")

    print("\n=== Final Perplexity Comparison ===")
    print(f"  RNN        : {rnn_ppls[-1]:.2f}")
    print(f"  LSTM       : {lstm_ppls[-1]:.2f}")
    print(f"  Transformer: {tf_ppls[-1]:.2f}")

    # Generate from "the"
    start_word = "the"
    start_id   = word_to_id.get(start_word, 0)
    gen_ids    = transformer_model.generate(start_id, length=8)
    gen_words  = [id_to_word.get(i, "?") for i in gen_ids]
    print(f"\nGenerated: {' '.join(gen_words)}")
