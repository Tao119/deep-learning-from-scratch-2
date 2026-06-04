"""
japanese_pos_tagging.py

Japanese Part-of-Speech Tagging with a character-level BiLSTM.

Tags: NOUN, VERB, PART, AUX, ADJ, ADV, PUNCT
      (名詞, 動詞, 助詞, 助動詞, 形容詞, 副詞, 記号)

Architecture:
  character embedding -> forward LSTM + backward LSTM (manually reversed)
  -> concatenate hidden states -> FC -> softmax -> cross-entropy

Uses TimeLSTM from common/time_layers.py.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.time_layers import LSTM, _sigmoid

np.random.seed(42)

# ---------------------------------------------------------------------------
# Tag set
# ---------------------------------------------------------------------------

TAGS = ["NOUN", "VERB", "PART", "AUX", "ADJ", "ADV", "PUNCT"]
tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for t, i in tag2id.items()}
NUM_TAGS = len(TAGS)

# ---------------------------------------------------------------------------
# Synthetic training corpus – 200 sentences
# ---------------------------------------------------------------------------

_NOUNS  = ["猫", "犬", "鳥", "魚", "花", "木", "山", "川", "空", "海",
           "子供", "先生", "学生", "医者", "警察", "電車", "車", "船", "飛行機",
           "りんご", "みかん", "バナナ", "本", "映画", "音楽", "友達", "家族",
           "会社", "学校", "病院", "公園", "駅", "店", "道路", "建物"]
_VERBS  = ["走る", "泳ぐ", "飛ぶ", "食べる", "飲む", "読む", "書く", "見る",
           "聞く", "話す", "歩く", "止まる", "起きる", "寝る", "働く", "遊ぶ",
           "来る", "行く", "帰る", "買う", "売る", "作る", "壊す", "直す"]
_ADJS   = ["速い", "遅い", "大きい", "小さい", "赤い", "青い", "白い", "黒い",
           "美しい", "醜い", "新しい", "古い", "高い", "低い", "長い", "短い"]
_ADVS   = ["とても", "少し", "ゆっくり", "急いで", "静かに", "元気に", "きれいに"]
_PARTS  = ["が", "を", "に", "で", "と", "から", "まで", "も", "は", "の"]
_AUXS   = ["です", "ます", "た", "ない", "ている", "られる", "させる"]
_PUNCT  = ["。", "、", "！", "？"]

# Template patterns: each is a list of (word_fn, tag) pairs
# word_fn is called with the rng to produce a word

def _make_sentence(rng):
    """Produce a (tokens, tags) pair from one of several templates."""
    pattern = rng.integers(0, 8)

    def noun():  return _NOUNS[rng.integers(len(_NOUNS))]
    def verb():  return _VERBS[rng.integers(len(_VERBS))]
    def adj():   return _ADJS[rng.integers(len(_ADJS))]
    def adv():   return _ADVS[rng.integers(len(_ADVS))]
    def part():  return _PARTS[rng.integers(len(_PARTS))]
    def aux():   return _AUXS[rng.integers(len(_AUXS))]
    def punct(): return _PUNCT[rng.integers(len(_PUNCT))]

    templates = [
        # NOUN が VERB 。
        lambda: ([noun(), "が", verb(), "。"],
                 ["NOUN", "PART", "VERB", "PUNCT"]),
        # NOUN は ADJ です 。
        lambda: ([noun(), "は", adj(), "です", "。"],
                 ["NOUN", "PART", "ADJ", "AUX", "PUNCT"]),
        # ADV VERB 。
        lambda: ([adv(), verb(), "。"],
                 ["ADV", "VERB", "PUNCT"]),
        # NOUN が NOUN を VERB 。
        lambda: ([noun(), "が", noun(), "を", verb(), "。"],
                 ["NOUN", "PART", "NOUN", "PART", "VERB", "PUNCT"]),
        # NOUN は ADV VERB 。
        lambda: ([noun(), "は", adv(), verb(), "。"],
                 ["NOUN", "PART", "ADV", "VERB", "PUNCT"]),
        # ADJ NOUN が VERB た 。
        lambda: ([adj(), noun(), "が", verb(), "た", "。"],
                 ["ADJ", "NOUN", "PART", "VERB", "AUX", "PUNCT"]),
        # NOUN は NOUN に VERB た 。
        lambda: ([noun(), "は", noun(), "に", verb(), "た", "。"],
                 ["NOUN", "PART", "NOUN", "PART", "VERB", "AUX", "PUNCT"]),
        # NOUN は ADJ NOUN です 。
        lambda: ([noun(), "は", adj(), noun(), "です", "。"],
                 ["NOUN", "PART", "ADJ", "NOUN", "AUX", "PUNCT"]),
    ]
    words, tags = templates[pattern]()
    return words, tags


def make_corpus(n=200):
    rng = np.random.default_rng(0)
    sentences, tag_seqs = [], []
    for _ in range(n):
        words, tags = _make_sentence(rng)
        sentences.append(words)
        tag_seqs.append(tags)
    return sentences, tag_seqs


# ---------------------------------------------------------------------------
# Character-level vocabulary
# ---------------------------------------------------------------------------

def build_char_vocab(sentences):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent in sentences:
        for word in sent:
            for ch in word:
                if ch not in vocab:
                    vocab[ch] = len(vocab)
    return vocab


def tokenize_chars(sentences, tag_seqs, char_vocab, max_len):
    """Convert each word to its first character id. Pad to max_len."""
    N = len(sentences)
    xs = np.zeros((N, max_len), dtype=np.int32)
    ys = np.full((N, max_len), -1, dtype=np.int32)
    lengths = []
    for i, (sent, tags) in enumerate(zip(sentences, tag_seqs)):
        L = min(len(sent), max_len)
        lengths.append(L)
        for t in range(L):
            word = sent[t]
            # Use first character as token representation
            ch = word[0]
            xs[i, t] = char_vocab.get(ch, 1)
            ys[i, t] = tag2id[tags[t]]
    return xs, ys, lengths


# ---------------------------------------------------------------------------
# Embedding layer
# ---------------------------------------------------------------------------

class Embedding:
    def __init__(self, V, D):
        self.W = (np.random.randn(V, D) / np.sqrt(V)).astype("f")
        self.params = [self.W]
        self.grads = [np.zeros_like(self.W)]
        self.cache = None

    def forward(self, xs):
        self.cache = xs
        return self.W[xs]

    def backward(self, dout):
        dW = np.zeros_like(self.W)
        np.add.at(dW, self.cache.reshape(-1), dout.reshape(-1, dout.shape[-1]))
        self.grads[0][...] = dW


# ---------------------------------------------------------------------------
# Single LSTM cell wrapper
# ---------------------------------------------------------------------------

class LSTMCell:
    def __init__(self, D, H):
        scale = np.sqrt(D + H)
        Wx = (np.random.randn(D, 4 * H) / scale).astype("f")
        Wh = (np.random.randn(H, 4 * H) / scale).astype("f")
        b  = np.zeros(4 * H, dtype="f")
        self.cell = LSTM(Wx, Wh, b)
        self.params = self.cell.params
        self.grads  = self.cell.grads
        self.H = H

    def step(self, x_t, h_prev, c_prev):
        return self.cell.forward(x_t, h_prev, c_prev)


# ---------------------------------------------------------------------------
# BiLSTM layer using TimeLSTM-style LSTM cells
# ---------------------------------------------------------------------------

class BiLSTM:
    """
    Forward LSTM + Backward LSTM over a padded batch.
    Input : xs  (N, T, D)
    Output: out (N, T, 2H)
    """

    def __init__(self, D, H):
        self.H   = H
        self.fwd = LSTMCell(D, H)
        self.bwd = LSTMCell(D, H)
        self.params = self.fwd.params + self.bwd.params
        self.grads  = self.fwd.grads  + self.bwd.grads
        self.cache  = None

    def forward(self, xs, lengths):
        N, T, D = xs.shape
        H = self.H

        fwd_hs = np.zeros((N, T, H), dtype="f")
        h_f = np.zeros((N, H), dtype="f")
        c_f = np.zeros((N, H), dtype="f")
        fwd_cache = []
        for t in range(T):
            h_f, c_f = self.fwd.step(xs[:, t, :], h_f, c_f)
            fwd_hs[:, t, :] = h_f
            fwd_cache.append(self.fwd.cell.cache)

        bwd_hs = np.zeros((N, T, H), dtype="f")
        h_b = np.zeros((N, H), dtype="f")
        c_b = np.zeros((N, H), dtype="f")
        bwd_cache = []
        for t in reversed(range(T)):
            h_b, c_b = self.bwd.step(xs[:, t, :], h_b, c_b)
            bwd_hs[:, t, :] = h_b
            bwd_cache.insert(0, self.bwd.cell.cache)

        out = np.concatenate([fwd_hs, bwd_hs], axis=2)
        self.cache = (xs, lengths, fwd_cache, bwd_cache)
        return out

    def backward(self, dout):
        xs, lengths, fwd_cache, bwd_cache = self.cache
        N, T, D = xs.shape
        H = self.H

        dfwd = dout[:, :, :H]
        dbwd = dout[:, :, H:]
        dxs  = np.zeros_like(xs)

        dh_f = np.zeros((N, H), dtype="f")
        dc_f = np.zeros((N, H), dtype="f")
        dWx_f = np.zeros_like(self.fwd.params[0])
        dWh_f = np.zeros_like(self.fwd.params[1])
        db_f  = np.zeros_like(self.fwd.params[2])
        for t in reversed(range(T)):
            self.fwd.cell.cache = fwd_cache[t]
            dx_t, dh_f, dc_f = self.fwd.cell.backward(dfwd[:, t, :] + dh_f, dc_f)
            dxs[:, t, :] += dx_t
            dWx_f += self.fwd.grads[0]
            dWh_f += self.fwd.grads[1]
            db_f  += self.fwd.grads[2]
        self.fwd.grads[0][...] = dWx_f
        self.fwd.grads[1][...] = dWh_f
        self.fwd.grads[2][...] = db_f

        dh_b = np.zeros((N, H), dtype="f")
        dc_b = np.zeros((N, H), dtype="f")
        dWx_b = np.zeros_like(self.bwd.params[0])
        dWh_b = np.zeros_like(self.bwd.params[1])
        db_b  = np.zeros_like(self.bwd.params[2])
        for t in range(T):
            self.bwd.cell.cache = bwd_cache[t]
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
# Linear projection
# ---------------------------------------------------------------------------

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = (np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)).astype("f")
        self.b = np.zeros(out_dim, dtype="f")
        self.params = [self.W, self.b]
        self.grads  = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.cache  = None

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        x  = self.cache
        xr = x.reshape(-1, x.shape[-1])
        dr = dout.reshape(-1, dout.shape[-1])
        self.grads[0][...] = xr.T @ dr
        self.grads[1][...] = dr.sum(axis=0)
        return (dr @ self.W.T).reshape(x.shape)


# ---------------------------------------------------------------------------
# BiLSTM POS Tagger model
# ---------------------------------------------------------------------------

class BiLSTMPOSTagger:
    def __init__(self, vocab_size, emb_dim, hidden_size, num_tags):
        self.embedding = Embedding(vocab_size, emb_dim)
        self.bilstm    = BiLSTM(emb_dim, hidden_size)
        self.linear    = Linear(2 * hidden_size, num_tags)
        self.num_tags  = num_tags
        self.params = (self.embedding.params
                       + self.bilstm.params
                       + self.linear.params)
        self.grads  = (self.embedding.grads
                       + self.bilstm.grads
                       + self.linear.grads)
        self._cache = None

    def forward(self, xs, ys, lengths):
        emb    = self.embedding.forward(xs)
        bi_out = self.bilstm.forward(emb, lengths)
        logits = self.linear.forward(bi_out)

        N, T, L = logits.shape
        flat_logits = logits.reshape(N * T, L)
        flat_labels = ys.reshape(N * T)

        mask = flat_labels >= 0
        ml   = flat_logits[mask]
        tl   = flat_labels[mask]

        ml_max = ml.max(axis=1, keepdims=True)
        exp_ml = np.exp(ml - ml_max)
        probs  = exp_ml / exp_ml.sum(axis=1, keepdims=True)
        loss   = -np.log(probs[np.arange(len(tl)), tl] + 1e-7).mean()

        self._cache = (logits, ys, probs, tl, mask, N, T, L)
        return loss

    def backward(self):
        logits, ys, probs, tl, mask, N, T, L = self._cache
        d_probs = probs.copy()
        d_probs[np.arange(len(tl)), tl] -= 1
        d_probs /= len(tl)

        flat_labels = ys.reshape(N * T)
        d_flat = np.zeros((N * T, L), dtype="f")
        d_flat[mask] = d_probs
        d_logits = d_flat.reshape(N, T, L)

        d_bi  = self.linear.backward(d_logits)
        d_emb = self.bilstm.backward(d_bi)
        self.embedding.backward(d_emb)

    def predict(self, xs, lengths):
        emb    = self.embedding.forward(xs)
        bi_out = self.bilstm.forward(emb, lengths)
        logits = self.linear.forward(bi_out)
        preds  = logits.argmax(axis=2)
        return [list(preds[i, :lengths[i]]) for i in range(len(lengths))]


# ---------------------------------------------------------------------------
# Adam optimiser
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr_t = self.lr * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g ** 2
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_epochs=100, batch_size=16):
    sentences, tag_seqs = make_corpus(n=200)
    char_vocab = build_char_vocab(sentences)
    max_len = max(len(s) for s in sentences)

    xs, ys, lengths = tokenize_chars(sentences, tag_seqs, char_vocab, max_len)

    vocab_size  = len(char_vocab)
    emb_dim     = 16
    hidden_size = 32

    model     = BiLSTMPOSTagger(vocab_size, emb_dim, hidden_size, NUM_TAGS)
    optimizer = Adam(lr=5e-3)

    N = len(xs)
    print("=" * 55)
    print("BiLSTM Japanese POS Tagger")
    print(f"  sentences={N}  vocab={vocab_size}  tags={NUM_TAGS}")
    print(f"  tags: {TAGS}")
    print("=" * 55)

    for epoch in range(1, n_epochs + 1):
        idx   = np.random.permutation(N)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, N, batch_size):
            b_idx = idx[start:start + batch_size]
            xb = xs[b_idx]
            yb = ys[b_idx]
            lb = [lengths[i] for i in b_idx]
            loss = model.forward(xb, yb, lb)
            model.backward()
            total_norm = np.sqrt(sum((g ** 2).sum() for g in model.grads))
            if total_norm > 5.0:
                for g in model.grads:
                    g *= 5.0 / (total_norm + 1e-8)
            optimizer.update(model.params, model.grads)
            total_loss += loss
            n_batches  += 1

        if epoch % 10 == 0:
            all_preds = model.predict(xs, lengths)
            correct = total = 0
            for i, (preds, L) in enumerate(zip(all_preds, lengths)):
                for t in range(L):
                    if ys[i, t] >= 0:
                        total += 1
                        if preds[t] == ys[i, t]:
                            correct += 1
            acc = correct / total if total else 0.0
            print(f"epoch {epoch:>3}  loss={total_loss / n_batches:.4f}  acc={acc*100:.1f}%")

    return model, sentences, tag_seqs, xs, ys, lengths, char_vocab


def demo(model, sentences, tag_seqs, xs, ys, lengths):
    """Tag 5 new sentences not in training (constructed from known vocab)."""
    print("\n" + "=" * 55)
    print("Demo: POS-tag 5 example sentences")
    print("=" * 55)

    demo_sentences = [
        (["猫", "が", "速い", "。"],         ["NOUN", "PART", "ADJ",  "PUNCT"]),
        (["先生", "は", "本", "を", "読む", "た", "。"], ["NOUN", "PART", "NOUN", "PART", "VERB", "AUX", "PUNCT"]),
        (["鳥", "は", "とても", "美しい", "。"],  ["NOUN", "PART", "ADV",  "ADJ",  "PUNCT"]),
        (["子供", "が", "公園", "で", "遊ぶ", "。"], ["NOUN", "PART", "NOUN", "PART", "VERB", "PUNCT"]),
        (["電車", "は", "新しい", "です", "。"],   ["NOUN", "PART", "ADJ",  "AUX",  "PUNCT"]),
    ]

    # Build a minimal char vocab from training sentences to handle demo words
    char_vocab_demo = build_char_vocab([s for s, _ in demo_sentences])
    # Merge with existing (use model's trained vocab via the xs)
    # We re-use the global char_vocab that was built during training
    global _GLOBAL_CHAR_VOCAB
    for ch, idx in char_vocab_demo.items():
        if ch not in _GLOBAL_CHAR_VOCAB:
            pass  # unknown chars map to <UNK>=1

    max_demo_len = max(len(s) for s, _ in demo_sentences)
    N_d = len(demo_sentences)
    xd = np.zeros((N_d, max_demo_len), dtype=np.int32)
    yd_true = []
    lens_d  = []
    for i, (words, tags) in enumerate(demo_sentences):
        L = len(words)
        lens_d.append(L)
        yd_true.append(tags)
        for t, word in enumerate(words):
            ch = word[0]
            xd[i, t] = _GLOBAL_CHAR_VOCAB.get(ch, 1)

    preds = model.predict(xd, lens_d)

    for i, (words, true_tags) in enumerate(demo_sentences):
        pred_tags = [id2tag[p] for p in preds[i]]
        print(f"\nSentence: {' '.join(words)}")
        print(f"  True: {true_tags}")
        print(f"  Pred: {pred_tags}")
        matches = sum(p == t for p, t in zip(pred_tags, true_tags))
        print(f"  Acc:  {matches}/{len(true_tags)} = {matches/len(true_tags)*100:.0f}%")


_GLOBAL_CHAR_VOCAB = {}

if __name__ == "__main__":
    model, sentences, tag_seqs, xs, ys, lengths, char_vocab = train(n_epochs=100, batch_size=16)
    _GLOBAL_CHAR_VOCAB = char_vocab

    # Final accuracy
    all_preds = model.predict(xs, lengths)
    correct = total = 0
    for i, (preds, L) in enumerate(zip(all_preds, lengths)):
        for t in range(L):
            if ys[i, t] >= 0:
                total += 1
                if preds[t] == ys[i, t]:
                    correct += 1
    final_acc = correct / total if total else 0.0
    print(f"\nFinal token accuracy: {final_acc*100:.1f}%")

    demo(model, sentences, tag_seqs, xs, ys, lengths)
