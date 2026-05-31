"""
BERT-style Masked Language Model (MLM) pretraining — pure NumPy.

Architecture
------------
  token embedding + positional encoding
  → 2 x BidirectionalTransformerBlock (no causal mask, full attention)
  → MLM head: LayerNorm(32→32) → Affine(32→vocab_size)

Hyper-parameters
  d_model=32, n_heads=2, n_layers=2, d_ff=64
  MASK=0, PAD=1  (special token ids)
  mask_prob=0.15
  500 epochs, batch = all sentences (batch-all)

Corpus: 100 synthetic English sentences from templates:
  subject {animal} verb at/in {place} .
  e.g. "the dog runs in the park ."

Evaluation: 5 fill-in-the-blank sentences, top-3 predictions per [MASK]
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.optimizer import Adam


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy_flat(logits, targets):
    """
    logits : (M, V)  — only masked positions
    targets: (M,)    — original token ids
    """
    probs = _softmax(logits)
    M = len(targets)
    loss = -np.log(probs[np.arange(M), targets] + 1e-7).mean()
    return loss, probs


def clip_grads(grads, max_norm):
    total = sum(np.sum(g ** 2) for g in grads)
    norm = np.sqrt(total)
    if norm > max_norm:
        rate = max_norm / (norm + 1e-6)
        for g in grads:
            g *= rate


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

def build_corpus():
    """
    Return exactly 100 synthetic English sentences from three templates.

    Template A (animals × verbs × places): subject verb prep place .
    Template B (jobs × workplaces):        "the teacher works at school ."
    Template C (animals × adjectives):     "the big dog runs in the park ."
    """
    animals   = ["dog", "cat", "bird", "fox", "rabbit",
                 "bear", "deer", "wolf", "lion", "eagle"]
    verbs_loc = [
        ("runs",   "in",   "the park"),
        ("sleeps", "at",   "home"),
        ("eats",   "near", "the river"),
        ("plays",  "in",   "the meadow"),
        ("jumps",  "over", "the stream"),
        ("rests",  "by",   "the lake"),
        ("hides",  "in",   "the forest"),
        ("hunts",  "in",   "the mountains"),
        ("swims",  "in",   "the pond"),
        ("flies",  "above","the clouds"),
    ]

    sentences = []

    # --- Template A: 10 animals × 5 verb-location combos = 50 sentences ---
    for animal in animals:
        for verb, prep, place in verbs_loc[:5]:
            sentences.append(f"the {animal} {verb} {prep} {place} .")

    # --- Template B: jobs × workplaces = 25 sentences ---
    jobs   = ["teacher", "doctor", "farmer", "driver", "cook"]
    places_b = ["school", "the hospital", "the farm", "the road", "the kitchen"]
    for job, place in zip(jobs, places_b):
        for verb in ["works", "lives", "rests", "eats", "sleeps"]:
            sentences.append(f"the {job} {verb} at {place} .")

    # --- Template C: adjective + animal × places = 25 sentences ---
    adjs    = ["big", "small", "fast", "young", "old"]
    animals_c = ["dog", "cat", "bird", "fox", "bear"]
    for adj, animal in zip(adjs, animals_c):
        for verb, prep, place in verbs_loc[5:]:
            sentences.append(f"the {adj} {animal} {verb} {prep} {place} .")

    return sentences[:100]


# ---------------------------------------------------------------------------
# Tokeniser  (word-level, MASK=0, PAD=1)
# ---------------------------------------------------------------------------

def build_vocab(sentences):
    """
    Special tokens: [MASK]=0, [PAD]=1
    All other words are assigned ids starting from 2.
    """
    all_words = []
    for sent in sentences:
        all_words.extend(sent.split())
    unique = sorted(set(all_words))
    vocab = ["[MASK]", "[PAD]"] + unique   # 0=MASK, 1=PAD
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


def tokenize(sentences, w2i, max_len):
    PAD = w2i["[PAD]"]
    data = []
    for sent in sentences:
        ids = [w2i.get(w, w2i["[PAD]"]) for w in sent.split()]
        if len(ids) < max_len:
            ids += [PAD] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        data.append(ids)
    return np.array(data, dtype=np.int32)


def apply_mlm_masking(tokens, mask_prob, w2i):
    """
    Replace mask_prob fraction of non-special tokens with [MASK] (id=0).
    Returns:
        masked : same shape, with some positions set to MASK id
        labels : same shape, -1 where no prediction needed, original id at masked spots
    """
    MASK = w2i["[MASK]"]
    PAD  = w2i["[PAD]"]

    masked = tokens.copy()
    labels = np.full_like(tokens, -1)

    candidate = tokens != PAD   # never mask [PAD]
    rng = np.random.random(tokens.shape)
    selected = candidate & (rng < mask_prob)

    masked[selected] = MASK
    labels[selected] = tokens[selected]
    return masked, labels


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps   = eps
        self.gamma = np.ones(d_model, dtype="f")
        self.beta  = np.zeros(d_model, dtype="f")
        self.params = [self.gamma, self.beta]
        self.grads  = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]
        self.cache  = None

    def forward(self, x):
        mu   = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1,  keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        out  = self.gamma * xhat + self.beta
        self.cache = (x, xhat, mu, var)
        return out

    def backward(self, dout):
        x, xhat, mu, var = self.cache
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dgamma  = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
        dbeta   = dout.sum(axis=tuple(range(dout.ndim - 1)))
        dxhat   = dout * self.gamma
        dx = std_inv * (dxhat
                        - dxhat.mean(axis=-1, keepdims=True)
                        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta
        return dx


# ---------------------------------------------------------------------------
# Full (bidirectional) multi-head self-attention
# ---------------------------------------------------------------------------

class FullSelfAttention:
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

    def _split(self, x):
        N, T, _ = x.shape
        return x.reshape(N, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge(self, x):
        N, h, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, T, h * dh)

    def forward(self, x, pad_mask=None):
        N, T, _ = x.shape
        Q = self._split(x @ self.Wq)
        K = self._split(x @ self.Wk)
        V = self._split(x @ self.Wv)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)

        if pad_mask is not None:
            scores = scores + pad_mask[:, np.newaxis, np.newaxis, :] * (-1e9)

        A   = _softmax(scores)
        out = self._merge(A @ V) @ self.Wo

        self.cache = (x, Q, K, V, A)
        return out

    def backward(self, dout):
        x, Q, K, V, A = self.cache
        N, T, _ = x.shape

        out_pre = self._merge(A @ V)
        dWo = out_pre.reshape(N * T, self.d_model).T @ dout.reshape(N * T, self.d_model)
        d_merged = dout @ self.Wo.T

        d_AV = self._split(d_merged)
        dA   = d_AV @ V.transpose(0, 1, 3, 2)
        dV   = A.transpose(0, 1, 3, 2) @ d_AV

        dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dscores /= np.sqrt(self.d_head)

        dQ = dscores @ K
        dK = dscores.transpose(0, 1, 3, 2) @ Q

        dQ_m = self._merge(dQ)
        dK_m = self._merge(dK)
        dV_m = self._merge(dV)

        dWq = x.reshape(N * T, self.d_model).T @ dQ_m.reshape(N * T, self.d_model)
        dWk = x.reshape(N * T, self.d_model).T @ dK_m.reshape(N * T, self.d_model)
        dWv = x.reshape(N * T, self.d_model).T @ dV_m.reshape(N * T, self.d_model)

        dx = dQ_m @ self.Wq.T + dK_m @ self.Wk.T + dV_m @ self.Wv.T

        self.grads[0][...] = dWq
        self.grads[1][...] = dWk
        self.grads[2][...] = dWv
        self.grads[3][...] = dWo
        return dx


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

class FFN:
    def __init__(self, d_model, d_ff):
        scale = np.sqrt(d_model)
        self.W1 = (np.random.randn(d_model, d_ff) / scale).astype("f")
        self.b1 = np.zeros(d_ff, dtype="f")
        self.W2 = (np.random.randn(d_ff, d_model) / np.sqrt(d_ff)).astype("f")
        self.b2 = np.zeros(d_model, dtype="f")
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.grads  = [np.zeros_like(p) for p in self.params]
        self.cache  = None

    def forward(self, x):
        h   = np.maximum(0, x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        xr = x.reshape(-1, x.shape[-1])
        hr = h.reshape(-1, h.shape[-1])
        dr = dout.reshape(-1, dout.shape[-1])
        dW2 = hr.T @ dr
        db2 = dr.sum(axis=0)
        dh  = dr @ self.W2.T
        dh[hr == 0] = 0
        dW1 = xr.T @ dh
        db1 = dh.sum(axis=0)
        dx  = (dh @ self.W1.T).reshape(x.shape)
        self.grads[0][...] = dW1
        self.grads[1][...] = db1
        self.grads[2][...] = dW2
        self.grads[3][...] = db2
        return dx


# ---------------------------------------------------------------------------
# Transformer encoder block (bidirectional, pre-norm)
# ---------------------------------------------------------------------------

class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.norm1 = LayerNorm(d_model)
        self.attn  = FullSelfAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.ffn   = FFN(d_model, d_ff)
        self.params = (self.norm1.params + self.attn.params
                       + self.norm2.params + self.ffn.params)
        self.grads  = (self.norm1.grads  + self.attn.grads
                       + self.norm2.grads  + self.ffn.grads)
        self.cache  = None

    def forward(self, x, pad_mask=None):
        h   = x + self.attn.forward(self.norm1.forward(x), pad_mask)
        out = h + self.ffn.forward(self.norm2.forward(h))
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        dh_ffn = self.ffn.backward(self.norm2.backward(dout))
        dh     = dout + dh_ffn
        dx_att = self.attn.backward(self.norm1.backward(dh))
        return dh + dx_att


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def positional_encoding(max_len, d_model):
    pe  = np.zeros((max_len, d_model), dtype="f")
    pos = np.arange(max_len)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    if d_model > 1:
        pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe


# ---------------------------------------------------------------------------
# BERT MLM model
# ---------------------------------------------------------------------------

class BertMLM:
    """
    Bidirectional Transformer + MLM head.
    MLM head: LayerNorm(d_model) → Affine(d_model → vocab_size)
    """

    def __init__(self, vocab_size, d_model=32, n_heads=2, n_layers=2,
                 d_ff=64, max_len=32):
        self.d_model    = d_model
        self.vocab_size = vocab_size

        scale = np.sqrt(d_model)
        self.embed_W = (np.random.randn(vocab_size, d_model) / scale).astype("f")
        self.pe      = positional_encoding(max_len, d_model)

        self.blocks = [EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        # MLM head: LayerNorm → Affine
        self.head_norm = LayerNorm(d_model)
        self.head_W    = (np.random.randn(d_model, vocab_size) / scale).astype("f")
        self.head_b    = np.zeros(vocab_size, dtype="f")

        # Collect params/grads
        self.params = [self.embed_W, self.head_W, self.head_b]
        self.grads  = [np.zeros_like(self.embed_W),
                       np.zeros_like(self.head_W),
                       np.zeros_like(self.head_b)]
        for blk in self.blocks:
            self.params += list(blk.params)
            self.grads  += list(blk.grads)
        self.params += self.head_norm.params
        self.grads  += self.head_norm.grads

        self._cache = None

    def forward(self, xs, labels, pad_mask=None):
        """
        xs     : (N, T) masked token ids
        labels : (N, T) original ids at masked positions, -1 elsewhere
        """
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]
        self._emb_in = xs

        for blk in self.blocks:
            x = blk.forward(x, pad_mask)

        h      = self.head_norm.forward(x)
        logits = h @ self.head_W + self.head_b   # (N, T, V)

        mask_pos    = labels != -1               # (N, T) bool
        flat_logits = logits[mask_pos]           # (M, V)
        flat_labels = labels[mask_pos]           # (M,)

        if len(flat_labels) == 0:
            self._cache = (x, h, logits, mask_pos, flat_labels, None)
            return 0.0

        loss, probs = _cross_entropy_flat(flat_logits, flat_labels)
        self._cache = (x, h, logits, mask_pos, flat_labels, probs)
        return loss

    def backward(self, dout=1.0):
        x, h, logits, mask_pos, flat_labels, probs = self._cache
        if probs is None:
            return
        N, T, _ = logits.shape
        M = len(flat_labels)

        # Gradient of loss w.r.t. flat_logits
        d_flat = probs.copy()
        d_flat[np.arange(M), flat_labels] -= 1
        d_flat *= dout / M

        # Scatter back to (N, T, V)
        dlogits = np.zeros_like(logits)
        dlogits[mask_pos] = d_flat

        # MLM head backward
        dh = dlogits @ self.head_W.T
        self.grads[2][...] = dlogits.sum(axis=(0, 1))
        self.grads[1][...] = (
            h.reshape(N * T, self.d_model).T
            @ dlogits.reshape(N * T, self.vocab_size)
        )

        dx = self.head_norm.backward(dh)

        for blk in reversed(self.blocks):
            dx = blk.backward(dx)

        # Embedding gradient
        dW_emb = np.zeros_like(self.embed_W)
        np.add.at(dW_emb, self._emb_in.reshape(-1),
                  dx.reshape(N * T, self.d_model))
        self.grads[0][...] = dW_emb

    def predict_top_k(self, xs, positions, k=3, pad_mask=None):
        """
        xs        : (1, T) masked token ids
        positions : list of (row, col) tuples of masked positions
        Returns list of (original_pos, top_k_ids)
        """
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]
        for blk in self.blocks:
            x = blk.forward(x, pad_mask)
        h      = self.head_norm.forward(x)
        logits = h @ self.head_W + self.head_b   # (N, T, V)
        results = []
        for (row, col) in positions:
            top_k = logits[row, col].argsort()[::-1][:k].tolist()
            results.append((col, top_k))
        return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, tokens, w2i, epochs=500, mask_prob=0.15, lr=1e-3):
    optimizer = Adam(lr=lr)
    PAD = w2i["[PAD]"]

    for epoch in range(1, epochs + 1):
        # Apply masking to the entire corpus (batch = all)
        masked, labels = apply_mlm_masking(tokens, mask_prob, w2i)
        pad_mask = (masked == PAD).astype("f")

        # Skip epoch if nothing was masked (extremely rare)
        if not (labels != -1).any():
            continue

        loss = model.forward(masked, labels, pad_mask)
        model.backward()
        clip_grads(model.grads, 1.0)
        optimizer.update(model.params, model.grads)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  MLM loss = {loss:.4f}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_accuracy(model, tokens, w2i, mask_prob=0.15, n_trials=5):
    """
    Estimate top-1 accuracy on masked tokens by running several masking trials.
    """
    PAD = w2i["[PAD]"]
    correct = 0
    total   = 0

    for _ in range(n_trials):
        masked, labels = apply_mlm_masking(tokens, mask_prob, w2i)
        pad_mask = (masked == PAD).astype("f")
        if not (labels != -1).any():
            continue

        N, T = masked.shape
        x = model.embed_W[masked] + model.pe[:T]
        for blk in model.blocks:
            x = blk.forward(x, pad_mask)
        h      = model.head_norm.forward(x)
        logits = h @ model.head_W + model.head_b

        mask_pos    = labels != -1
        flat_logits = logits[mask_pos]
        flat_labels = labels[mask_pos]

        preds = flat_logits.argmax(axis=1)
        correct += int((preds == flat_labels).sum())
        total   += len(flat_labels)

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


def demo_fill_mask(model, w2i, i2w, max_len):
    MASK = w2i["[MASK]"]
    PAD  = w2i["[PAD]"]

    test_sents = [
        ("the dog runs in the park .",   2),   # mask 'dog'
        ("the cat sleeps at home .",     3),   # mask 'sleeps'
        ("a bird eats near the river .", 1),   # mask 'bird'
        ("the fox plays in the field .", 4),   # mask 'plays'
        ("the wolf jumps over the sky .",2),   # mask 'wolf'
    ]

    print("\n" + "=" * 60)
    print("Fill-mask Demo  (top-3 predictions per [MASK])")
    print("=" * 60)

    n_correct = 0
    n_total   = 0

    for sent, mask_word_pos in test_sents:
        words  = sent.split()
        ids    = [w2i.get(w, PAD) for w in words]
        original_id = ids[mask_word_pos]
        original_w  = i2w.get(original_id, "[UNK]")

        masked_ids = ids.copy()
        masked_ids[mask_word_pos] = MASK

        # Pad to max_len
        if len(masked_ids) < max_len:
            masked_ids += [PAD] * (max_len - len(masked_ids))
        xs = np.array([masked_ids[:max_len]], dtype=np.int32)
        pad_mask = (xs == PAD).astype("f")

        results = model.predict_top_k(xs, [(0, mask_word_pos)], k=3, pad_mask=pad_mask)
        _, top3_ids = results[0]
        top3_words  = [i2w.get(tid, "[UNK]") for tid in top3_ids]

        display = words.copy()
        display[mask_word_pos] = "[MASK]"
        print(f"\n  Input    : {' '.join(display)}")
        print(f"  Original : '{original_w}'")
        print(f"  Top-3    : {top3_words}")

        if original_id in top3_ids:
            n_correct += 1
        n_total += 1

    print(f"\nTop-3 accuracy on demo sentences: {n_correct}/{n_total}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    np.random.seed(42)
    t0 = time.time()

    sentences = build_corpus()
    print(f"Corpus: {len(sentences)} sentences")
    print(f"Example: '{sentences[0]}'")

    w2i, i2w = build_vocab(sentences)
    vocab_size = len(w2i)
    print(f"Vocab size: {vocab_size}  (MASK=0, PAD=1)")

    # Determine max sentence length
    max_len = max(len(s.split()) for s in sentences) + 2   # small buffer
    tokens  = tokenize(sentences, w2i, max_len)
    print(f"Token matrix shape: {tokens.shape}")

    # Build and train model
    model = BertMLM(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_len=max_len,
    )

    print("\n--- Training ---")
    train(model, tokens, w2i, epochs=500, mask_prob=0.15, lr=1e-3)

    elapsed = time.time() - t0
    print(f"\nRuntime: {elapsed:.1f}s")

    # Accuracy on training corpus
    acc, correct, total = evaluate_accuracy(model, tokens, w2i,
                                            mask_prob=0.15, n_trials=10)
    print(f"\nMasked token top-1 accuracy: {correct}/{total} = {acc*100:.1f}%")

    # Fill-mask demo
    demo_fill_mask(model, w2i, i2w, max_len)
