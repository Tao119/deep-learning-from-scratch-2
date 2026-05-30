"""
BiLSTM NER tagger — pure NumPy implementation.

Dataset
-------
100 synthetic sentences built from ASCII-friendly tokens representing
Japanese-style entities (persons, places, organisations).

Model
-----
  Token embedding
  → BiLSTM  (forward LSTM + backward LSTM, hidden states concatenated)
  → Linear  (2H → num_labels)
  → CRF-style Viterbi decoding at inference
    (training uses plain cross-entropy on the per-token logits)
"""

import sys
sys.path.append("..")
import numpy as np
from common.time_layers import LSTM, _sigmoid


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

PERSONS = ["Tanaka", "Yamada", "Sato"]
PLACES  = ["Tokyo",  "Osaka",  "Kyoto"]
ORGS    = ["Sony",   "Toyota", "Honda"]
OTHER   = ["is", "in", "works", "at", "the", "a"]

TEMPLATES = [
    ("{PER} works at {ORG} in {LOC} .", ["B-PER", "O",     "O",     "B-ORG", "O",     "B-LOC", "O"]),
    ("{PER} is at {LOC} .",             ["B-PER", "O",     "O",     "B-LOC", "O"]),
    ("{PER} works at {ORG} .",          ["B-PER", "O",     "O",     "B-ORG", "O"]),
    ("{PER} is in {LOC} .",             ["B-PER", "O",     "O",     "B-LOC", "O"]),
    ("the {ORG} is in {LOC} .",         ["O",     "B-ORG", "O",     "O",     "B-LOC", "O"]),
    ("{PER} is at {ORG} .",             ["B-PER", "O",     "O",     "B-ORG", "O"]),
]

LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
label_to_id = {l: i for i, l in enumerate(LABEL_LIST)}
id_to_label = {i: l for l, i in label_to_id.items()}
NUM_LABELS  = len(LABEL_LIST)


def make_dataset(n=100):
    rng = np.random.default_rng(0)
    sentences, label_seqs = [], []
    for _ in range(n):
        tmpl_text, tmpl_labels = TEMPLATES[rng.integers(len(TEMPLATES))]
        per = PERSONS[rng.integers(len(PERSONS))]
        loc = PLACES[rng.integers(len(PLACES))]
        org = ORGS[rng.integers(len(ORGS))]
        text   = tmpl_text.replace("{PER}", per).replace("{LOC}", loc).replace("{ORG}", org)
        tokens = text.split()
        labels = list(tmpl_labels)  # same length as tokens
        sentences.append(tokens)
        label_seqs.append(labels)
    return sentences, label_seqs


def build_vocab(sentences):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent in sentences:
        for tok in sent:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode(sentences, label_seqs, vocab, max_len):
    N   = len(sentences)
    xs  = np.zeros((N, max_len), dtype=np.int32)
    ys  = np.full((N, max_len), -1, dtype=np.int32)   # -1 = pad (ignored in loss)
    lengths = []
    for i, (sent, labs) in enumerate(zip(sentences, label_seqs)):
        L = min(len(sent), max_len)
        lengths.append(L)
        for t in range(L):
            xs[i, t] = vocab.get(sent[t], 1)
            ys[i, t] = label_to_id[labs[t]]
    return xs, ys, lengths


# ---------------------------------------------------------------------------
# Single-step LSTM cell wrapper (stateless forward)
# ---------------------------------------------------------------------------

class LSTMCell:
    """Single LSTM step using the LSTM cell from common/time_layers.py."""

    def __init__(self, D, H):
        scale = np.sqrt(D + H)
        Wx = (np.random.randn(D, 4 * H) / scale).astype("f")
        Wh = (np.random.randn(H, 4 * H) / scale).astype("f")
        b  = np.zeros(4 * H, dtype="f")
        self.cell   = LSTM(Wx, Wh, b)
        self.params = self.cell.params
        self.grads  = self.cell.grads
        self.H      = H

    def step(self, x_t, h_prev, c_prev):
        return self.cell.forward(x_t, h_prev, c_prev)


# ---------------------------------------------------------------------------
# BiLSTM layer
# ---------------------------------------------------------------------------

class BiLSTM:
    """
    Forward + backward LSTM over a padded sequence.

    Input : xs (N, T, D)
    Output: (N, T, 2H)
    """

    def __init__(self, D, H):
        self.H       = H
        self.fwd     = LSTMCell(D, H)
        self.bwd     = LSTMCell(D, H)
        self.params  = self.fwd.params + self.bwd.params
        self.grads   = self.fwd.grads  + self.bwd.grads
        self.cache   = None

    def forward(self, xs, lengths):
        N, T, D = xs.shape
        H = self.H

        # Forward pass
        fwd_hs = np.zeros((N, T, H), dtype="f")
        h_f = np.zeros((N, H), dtype="f")
        c_f = np.zeros((N, H), dtype="f")
        fwd_cells_cache = []
        for t in range(T):
            h_f, c_f = self.fwd.step(xs[:, t, :], h_f, c_f)
            fwd_hs[:, t, :] = h_f
            fwd_cells_cache.append(self.fwd.cell.cache)

        # Backward pass (reverse in time)
        bwd_hs = np.zeros((N, T, H), dtype="f")
        h_b = np.zeros((N, H), dtype="f")
        c_b = np.zeros((N, H), dtype="f")
        bwd_cells_cache = []
        for t in reversed(range(T)):
            h_b, c_b = self.bwd.step(xs[:, t, :], h_b, c_b)
            bwd_hs[:, t, :] = h_b
            bwd_cells_cache.insert(0, self.bwd.cell.cache)

        out = np.concatenate([fwd_hs, bwd_hs], axis=2)   # (N, T, 2H)
        self.cache = (xs, lengths, fwd_cells_cache, bwd_cells_cache)
        return out

    def backward(self, dout):
        xs, lengths, fwd_cells_cache, bwd_cells_cache = self.cache
        N, T, D = xs.shape
        H = self.H

        dfwd = dout[:, :, :H]
        dbwd = dout[:, :, H:]

        dxs = np.zeros_like(xs)

        # Backward through forward LSTM
        dh_f = np.zeros((N, H), dtype="f")
        dc_f = np.zeros((N, H), dtype="f")
        dWx_f = np.zeros_like(self.fwd.params[0])
        dWh_f = np.zeros_like(self.fwd.params[1])
        db_f  = np.zeros_like(self.fwd.params[2])
        for t in reversed(range(T)):
            self.fwd.cell.cache = fwd_cells_cache[t]
            dx_t, dh_f, dc_f = self.fwd.cell.backward(dfwd[:, t, :] + dh_f, dc_f)
            dxs[:, t, :] += dx_t
            dWx_f += self.fwd.grads[0]
            dWh_f += self.fwd.grads[1]
            db_f  += self.fwd.grads[2]
        self.fwd.grads[0][...] = dWx_f
        self.fwd.grads[1][...] = dWh_f
        self.fwd.grads[2][...] = db_f

        # Backward through backward LSTM
        dh_b = np.zeros((N, H), dtype="f")
        dc_b = np.zeros((N, H), dtype="f")
        dWx_b = np.zeros_like(self.bwd.params[0])
        dWh_b = np.zeros_like(self.bwd.params[1])
        db_b  = np.zeros_like(self.bwd.params[2])
        for t in range(T):
            self.bwd.cell.cache = bwd_cells_cache[t]
            dx_t, dh_b, dc_b = self.bwd.cell.backward(dbwd[:, t, :] + dh_b, dc_b)
            dxs[:, t, :] += dx_t
            dWx_b += self.bwd.grads[0]
            dWh_b += self.bwd.grads[1]
            db_b  += self.bwd.grads[2]
        self.bwd.grads[0][...] = dWx_b
        self.bwd.grads[1][...] = dWh_b
        self.bwd.grads[2][...] = db_b

        return dxs


# ---------------------------------------------------------------------------
# Embedding layer
# ---------------------------------------------------------------------------

class Embedding:
    def __init__(self, V, D):
        self.W      = (np.random.randn(V, D) / np.sqrt(V)).astype("f")
        self.params = [self.W]
        self.grads  = [np.zeros_like(self.W)]
        self.cache  = None

    def forward(self, xs):
        self.cache = xs
        return self.W[xs]

    def backward(self, dout):
        dW = np.zeros_like(self.W)
        np.add.at(dW, self.cache.reshape(-1), dout.reshape(-1, dout.shape[-1]))
        self.grads[0][...] = dW


# ---------------------------------------------------------------------------
# Linear projection
# ---------------------------------------------------------------------------

class Linear:
    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(in_dim)
        self.W      = (np.random.randn(in_dim, out_dim) / scale).astype("f")
        self.b      = np.zeros(out_dim, dtype="f")
        self.params = [self.W, self.b]
        self.grads  = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.cache  = None

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        x = self.cache
        N_T = x.shape[0] * x.shape[1] if x.ndim == 3 else x.shape[0]
        xr   = x.reshape(-1, x.shape[-1])
        dr   = dout.reshape(-1, dout.shape[-1])
        dW   = xr.T @ dr
        db   = dr.sum(axis=0)
        dx   = (dr @ self.W.T).reshape(x.shape)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


# ---------------------------------------------------------------------------
# Viterbi decoder (simplified CRF)
# ---------------------------------------------------------------------------

class CRFViterbi:
    """
    Learnable transition matrix A (num_labels × num_labels).
    Training: cross-entropy on BiLSTM logits only (no CRF training).
    Inference: Viterbi decoding using emission (BiLSTM logits) + A.
    """

    def __init__(self, num_labels):
        self.num_labels = num_labels
        # Small random initialisation for transitions
        self.A      = (np.random.randn(num_labels, num_labels) * 0.01).astype("f")
        self.params = [self.A]
        self.grads  = [np.zeros_like(self.A)]

    def decode(self, logits_seq, length):
        """
        Viterbi decoding for a single sequence.

        Parameters
        ----------
        logits_seq : (T, num_labels)  emission scores (pre-softmax)
        length     : int, actual sequence length (rest is padding)

        Returns
        -------
        list of int, predicted label ids, length = length
        """
        T = length
        L = self.num_labels
        A = self.A

        # dp[t, j] = best score ending in label j at time t
        dp      = np.full((T, L), -np.inf, dtype="f")
        back    = np.zeros((T, L), dtype=np.int32)
        dp[0]   = logits_seq[0]
        for t in range(1, T):
            for j in range(L):
                scores    = dp[t - 1] + A[:, j] + logits_seq[t, j]
                best_prev = int(scores.argmax())
                dp[t, j]  = scores[best_prev]
                back[t, j] = best_prev

        # Traceback
        path = [int(dp[T - 1].argmax())]
        for t in reversed(range(1, T)):
            path.append(back[t, path[-1]])
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# BiLSTM-NER model
# ---------------------------------------------------------------------------

class BiLSTMNER:
    def __init__(self, vocab_size, emb_dim, hidden_size, num_labels):
        self.embedding = Embedding(vocab_size, emb_dim)
        self.bilstm    = BiLSTM(emb_dim, hidden_size)
        self.linear    = Linear(2 * hidden_size, num_labels)
        self.crf       = CRFViterbi(num_labels)
        self.num_labels = num_labels

        self.params = (self.embedding.params
                       + self.bilstm.params
                       + self.linear.params
                       + self.crf.params)
        self.grads  = (self.embedding.grads
                       + self.bilstm.grads
                       + self.linear.grads
                       + self.crf.grads)

    def forward(self, xs, ys, lengths):
        """
        xs      : (N, T) int token ids
        ys      : (N, T) int label ids, -1 for padding
        lengths : list of int, actual sentence lengths

        Returns cross-entropy loss over non-padding positions.
        """
        emb    = self.embedding.forward(xs)          # (N, T, D)
        bi_out = self.bilstm.forward(emb, lengths)   # (N, T, 2H)
        logits = self.linear.forward(bi_out)          # (N, T, L)

        N, T, L = logits.shape
        flat_logits = logits.reshape(N * T, L)
        flat_labels = ys.reshape(N * T)

        # Mask out padding positions (label == -1)
        mask    = flat_labels >= 0
        ml      = flat_logits[mask]
        tl      = flat_labels[mask]

        # Softmax + cross-entropy
        ml_max  = ml.max(axis=1, keepdims=True)
        exp_ml  = np.exp(ml - ml_max)
        probs   = exp_ml / exp_ml.sum(axis=1, keepdims=True)
        loss    = -np.log(probs[np.arange(len(tl)), tl] + 1e-7).mean()

        self.cache = (logits, ys, probs, tl, mask, N, T, L)
        return loss

    def backward(self):
        logits, ys, probs, tl, mask, N, T, L = self.cache

        # Gradient of cross-entropy w.r.t. masked logits
        d_probs = probs.copy()
        d_probs[np.arange(len(tl)), tl] -= 1
        d_probs /= len(tl)

        # Scatter back to full (N*T, L) shape
        flat_labels = ys.reshape(N * T)
        d_logits_flat = np.zeros((N * T, L), dtype="f")
        d_logits_flat[mask] = d_probs
        d_logits = d_logits_flat.reshape(N, T, L)

        d_bi   = self.linear.backward(d_logits)
        d_emb  = self.bilstm.backward(d_bi)
        self.embedding.backward(d_emb)

    def predict(self, xs, lengths):
        """Return predicted label id sequences (list of lists)."""
        emb    = self.embedding.forward(xs)
        bi_out = self.bilstm.forward(emb, lengths)
        logits = self.linear.forward(bi_out)          # (N, T, L)
        preds  = []
        for i, L_i in enumerate(lengths):
            seq = self.crf.decode(logits[i], L_i)
            preds.append(seq)
        return preds


# ---------------------------------------------------------------------------
# Adam optimiser (local copy to avoid circular imports)
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t     = 0
        self.m     = None
        self.v     = None

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr_t   = self.lr * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g ** 2
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Build dataset ----
    sentences, label_seqs = make_dataset(n=100)
    vocab = build_vocab(sentences)
    max_len = max(len(s) for s in sentences)

    xs, ys, lengths = encode(sentences, label_seqs, vocab, max_len)

    vocab_size  = len(vocab)
    emb_dim     = 16
    hidden_size = 32
    num_labels  = NUM_LABELS
    n_epochs    = 100
    batch_size  = 16

    model     = BiLSTMNER(vocab_size, emb_dim, hidden_size, num_labels)
    optimizer = Adam(lr=5e-3)

    N = len(xs)
    print("=" * 55)
    print("BiLSTM NER Training")
    print(f"  samples={N}  vocab={vocab_size}  labels={num_labels}")
    print("=" * 55)

    for epoch in range(1, n_epochs + 1):
        idx   = np.random.permutation(N)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, N, batch_size):
            b_idx = idx[start:start + batch_size]
            xb    = xs[b_idx]
            yb    = ys[b_idx]
            lb    = [lengths[i] for i in b_idx]
            loss  = model.forward(xb, yb, lb)
            model.backward()
            # Gradient clipping
            total_norm = np.sqrt(sum((g ** 2).sum() for g in model.grads))
            if total_norm > 5.0:
                for g in model.grads:
                    g *= 5.0 / (total_norm + 1e-8)
            optimizer.update(model.params, model.grads)
            total_loss += loss
            n_batches  += 1
        avg_loss = total_loss / n_batches
        if epoch % 10 == 0:
            # Token-level accuracy on full dataset
            all_preds = model.predict(xs, lengths)
            correct = total_tokens = 0
            for i, (preds, L) in enumerate(zip(all_preds, lengths)):
                for t in range(L):
                    if ys[i, t] >= 0:
                        total_tokens += 1
                        if preds[t] == ys[i, t]:
                            correct += 1
            acc = correct / total_tokens if total_tokens else 0.0
            print(f"epoch {epoch:>3}  loss={avg_loss:.4f}  token_acc={acc*100:.1f}%")

    # ---- Final evaluation ----
    all_preds = model.predict(xs, lengths)
    correct = total_tokens = 0
    for i, (preds, L) in enumerate(zip(all_preds, lengths)):
        for t in range(L):
            if ys[i, t] >= 0:
                total_tokens += 1
                if preds[t] == ys[i, t]:
                    correct += 1
    final_acc = correct / total_tokens if total_tokens else 0.0
    print(f"\nFinal token-level accuracy: {final_acc*100:.1f}%")

    # ---- Demo: predicted vs true tags for 3 sentences ----
    print("\n" + "=" * 55)
    print("Demo: predicted vs true tags (3 sentences)")
    print("=" * 55)
    demo_preds = model.predict(xs[:3], lengths[:3])
    for i in range(3):
        L    = lengths[i]
        toks = sentences[i][:L]
        true = [id_to_label[ys[i, t]] for t in range(L)]
        pred = [id_to_label[demo_preds[i][t]] for t in range(L)]
        print(f"\nSentence {i+1}: {' '.join(toks)}")
        print(f"  True : {true}")
        print(f"  Pred : {pred}")
