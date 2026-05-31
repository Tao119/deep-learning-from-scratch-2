"""
Attention-based Seq2Seq Translation — pure NumPy.
Japanese → English (synthetic data)

Japanese vocab : {田中,山田,鈴木,花子} {東京,大阪,京都,名古屋} {住む,働く,行く,来る} は に 。
English vocab  : {Tanaka,Yamada,Suzuki,Hanako} {Tokyo,Osaka,Kyoto,Nagoya}
                 {lives,works,goes,comes} in at .
Pattern        : "田中は東京に住む。" → "Tanaka lives in Tokyo ."

Tokenisation   : character-level Japanese, word-level English
Encoder        : TimeEmbedding(ja_vocab→32) → TimeLSTM(32→64, stateful=False)
Decoder        : TimeEmbedding(en_vocab→32) → TimeLSTM(32→64, stateful=True)
                 + additive attention over encoder hidden states
                 → concat(context[64], h_dec[64]) → Affine(128→en_vocab)
Training       : teacher-forcing, 300 epochs, batch=32
Evaluation     : BLEU-1 on 20 test pairs, 10-sentence demo
Output         : nlp_applications/experiments/02-seq2seq-translation/results.json
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.time_layers import TimeLSTM, TimeEmbedding, LSTM
from common.embedding import Embedding
from common.optimizer import Adam
from common.util import clip_grads


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

JA_NAMES  = ["田中", "山田", "鈴木", "花子"]
JA_CITIES = ["東京", "大阪", "京都", "名古屋"]
JA_VERBS  = ["住む", "働く", "行く", "来る"]

EN_NAMES  = ["Tanaka", "Yamada", "Suzuki", "Hanako"]
EN_CITIES = ["Tokyo", "Osaka", "Kyoto", "Nagoya"]
EN_VERBS  = ["lives", "works", "goes", "comes"]

# Verb → preposition mapping
VERB_PREP = {"lives": "in", "works": "at", "goes": "to", "comes": "to"}

BOS = "<BOS>"
EOS = "<EOS>"
PAD_EN = "<PAD>"


def build_dataset():
    """
    Generate all 64 (4×4×4) combinations, then replicate to ~200 pairs.
    """
    pairs = []
    for i, (jn, en) in enumerate(zip(JA_NAMES, EN_NAMES)):
        for j, (jc, ec) in enumerate(zip(JA_CITIES, EN_CITIES)):
            for k, (jv, ev) in enumerate(zip(JA_VERBS, EN_VERBS)):
                prep = VERB_PREP[ev]
                ja_sent = f"{jn}は{jc}に{jv}。"
                en_sent = f"{en} {ev} {prep} {ec} ."
                pairs.append((ja_sent, en_sent))

    # Replicate to get ~200 pairs, then shuffle
    rng = np.random.RandomState(42)
    while len(pairs) < 200:
        pairs.append(pairs[rng.randint(len(pairs))])
    pairs = pairs[:200]
    rng.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Tokenisers
# ---------------------------------------------------------------------------

def build_ja_vocab(pairs):
    """Character-level Japanese tokeniser."""
    chars = set()
    for ja, _ in pairs:
        chars.update(list(ja))
    chars = sorted(chars)
    w2i = {c: i for i, c in enumerate(chars)}
    i2w = {i: c for c, i in w2i.items()}
    return w2i, i2w


def build_en_vocab(pairs):
    """Word-level English tokeniser with BOS/EOS/PAD."""
    special = [PAD_EN, BOS, EOS]
    words = set()
    for _, en in pairs:
        words.update(en.split())
    words = sorted(words)
    vocab = special + words
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


def encode_ja(sent, w2i):
    return np.array([w2i[c] for c in sent], dtype=np.int32)


def encode_en(sent, w2i):
    words = sent.split()
    return np.array([w2i[BOS]] + [w2i[w] for w in words] + [w2i[EOS]],
                    dtype=np.int32)


def pad_sequences(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    out = np.full((len(seqs), max_len), pad_id, dtype=np.int32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


# ---------------------------------------------------------------------------
# Attention layer  (single step: query=h_dec[1,H], keys=hs_enc[N,T,H])
# ---------------------------------------------------------------------------

class StepAttention:
    """
    Dot-product attention at a single decoder time step.
    query : (N, H) — current decoder hidden state
    keys  : (N, T, H) — all encoder hidden states
    Returns context: (N, H)
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs_enc, h_dec):
        N, T, H = hs_enc.shape
        # scores: (N, T)
        hr = h_dec[:, np.newaxis, :].repeat(T, axis=1)  # (N, T, H)
        scores = (hs_enc * hr).sum(axis=2)               # (N, T)
        a = _softmax(scores)                              # (N, T)
        context = (hs_enc * a[:, :, np.newaxis]).sum(axis=1)  # (N, H)
        self.cache = (hs_enc, hr, a, h_dec)
        return context, a

    def backward(self, dcontext):
        hs_enc, hr, a, h_dec = self.cache
        N, T, H = hs_enc.shape

        # dcontext → da and dhs_enc (from weighted sum)
        da_raw = (dcontext[:, np.newaxis, :] * hs_enc).sum(axis=2)  # (N, T)
        dhs_enc_from_ctx = dcontext[:, np.newaxis, :] * a[:, :, np.newaxis]  # (N,T,H)

        # Softmax backward: dscores = a * (da_raw - (da_raw*a).sum(-1, keepdims))
        dscores = a * (da_raw - (da_raw * a).sum(axis=1, keepdims=True))  # (N, T)

        # dscores → dhs_enc (from dot product) and dhr (→ dh_dec)
        dhs_enc_from_score = dscores[:, :, np.newaxis] * hr      # (N, T, H)
        dhr = dscores[:, :, np.newaxis] * hs_enc                  # (N, T, H)
        dh_dec = dhr.sum(axis=1)                                   # (N, H)

        dhs_enc = dhs_enc_from_ctx + dhs_enc_from_score           # (N, T, H)
        return dhs_enc, dh_dec


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder:
    def __init__(self, ja_vocab_size, embed_dim=32, hidden_size=64):
        embed_W = (np.random.randn(ja_vocab_size, embed_dim) / 100).astype("f")
        Wx = (np.random.randn(embed_dim, 4 * hidden_size) / np.sqrt(embed_dim)).astype("f")
        Wh = (np.random.randn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)).astype("f")
        b  = np.zeros(4 * hidden_size, dtype="f")

        self.embed = TimeEmbedding(embed_W)
        self.lstm  = TimeLSTM(Wx, Wh, b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads  = self.embed.grads  + self.lstm.grads

    def forward(self, xs):
        emb = self.embed.forward(xs)       # (N, T, embed_dim)
        hs  = self.lstm.forward(emb)       # (N, T, H)
        return hs

    def backward(self, dhs):
        demb = self.lstm.backward(dhs)
        self.embed.backward(demb)


# ---------------------------------------------------------------------------
# Decoder with step-by-step attention (teacher forcing during training)
# ---------------------------------------------------------------------------

class AttentionDecoder:
    def __init__(self, en_vocab_size, embed_dim=32, hidden_size=64):
        self.hidden_size = hidden_size
        self.en_vocab_size = en_vocab_size

        embed_W = (np.random.randn(en_vocab_size, embed_dim) / 100).astype("f")
        Wx = (np.random.randn(embed_dim, 4 * hidden_size) / np.sqrt(embed_dim)).astype("f")
        Wh = (np.random.randn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)).astype("f")
        b  = np.zeros(4 * hidden_size, dtype="f")
        Wa = (np.random.randn(2 * hidden_size, en_vocab_size) / np.sqrt(2 * hidden_size)).astype("f")
        ba = np.zeros(en_vocab_size, dtype="f")

        self.embed = TimeEmbedding(embed_W)
        self.lstm  = TimeLSTM(Wx, Wh, b, stateful=True)
        self.affine_W = Wa
        self.affine_b = ba

        # Attention layers (one per decoder time step, created during forward)
        self._attn_layers = []

        self.params = (self.embed.params + self.lstm.params
                       + [self.affine_W, self.affine_b])
        self.grads  = (self.embed.grads  + self.lstm.grads
                       + [np.zeros_like(self.affine_W), np.zeros_like(self.affine_b)])

        self._cache = None

    def set_state(self, h, c):
        self.lstm.set_state(h, c)

    def forward(self, ys, hs_enc):
        """
        ys      : (N, T_dec) decoder input token ids (teacher forcing)
        hs_enc  : (N, T_enc, H) encoder hidden states
        Returns logits: (N, T_dec, en_vocab_size)
        """
        N, T_dec = ys.shape
        H = self.hidden_size

        emb = self.embed.forward(ys)    # (N, T_dec, embed_dim)
        hs_dec = self.lstm.forward(emb) # (N, T_dec, H)

        # Step-by-step attention
        self._attn_layers = []
        contexts = np.zeros((N, T_dec, H), dtype="f")
        for t in range(T_dec):
            attn = StepAttention()
            c_t, _ = attn.forward(hs_enc, hs_dec[:, t, :])   # (N, H)
            contexts[:, t, :] = c_t
            self._attn_layers.append(attn)

        # concat context + decoder hidden
        combined = np.concatenate([contexts, hs_dec], axis=2)  # (N, T, 2H)

        # Affine
        N_T = N * T_dec
        combined_2d = combined.reshape(N_T, 2 * H)
        logits_2d = combined_2d @ self.affine_W + self.affine_b  # (N_T, V)
        logits = logits_2d.reshape(N, T_dec, self.en_vocab_size)

        self._cache = (hs_enc, hs_dec, contexts, combined, combined_2d, emb)
        return logits

    def backward(self, dlogits):
        hs_enc, hs_dec, contexts, combined, combined_2d, emb = self._cache
        N, T_dec, _ = dlogits.shape
        H = self.hidden_size

        dlogits_2d = dlogits.reshape(N * T_dec, self.en_vocab_size)
        dcombined_2d = dlogits_2d @ self.affine_W.T
        self.grads[-2][...] = combined_2d.T @ dlogits_2d   # dWa
        self.grads[-1][...] = dlogits_2d.sum(axis=0)       # dba
        dcombined = dcombined_2d.reshape(N, T_dec, 2 * H)

        dcontexts  = dcombined[:, :, :H]
        dhs_dec    = dcombined[:, :, H:]

        # Backprop through attention layers
        dhs_enc_total = np.zeros_like(hs_enc)
        for t in reversed(range(T_dec)):
            attn = self._attn_layers[t]
            dhs_enc_t, dh_dec_t = attn.backward(dcontexts[:, t, :])
            dhs_enc_total += dhs_enc_t
            dhs_dec[:, t, :] += dh_dec_t

        # Backprop through LSTM
        demb = self.lstm.backward(dhs_dec)
        self.embed.backward(demb)
        return dhs_enc_total

    def generate(self, hs_enc, start_id, max_len, eos_id):
        """Greedy generation at inference time."""
        N = hs_enc.shape[0]
        H = self.hidden_size
        sampled = []
        inp = np.array([[start_id]] * N, dtype=np.int32)

        for _ in range(max_len):
            emb = self.embed.forward(inp)            # (N, 1, embed_dim)
            h_dec = self.lstm.forward(emb)           # (N, 1, H)

            attn = StepAttention()
            ctx, _ = attn.forward(hs_enc, h_dec[:, 0, :])   # (N, H)

            combined = np.concatenate([ctx, h_dec[:, 0, :]], axis=1)  # (N, 2H)
            logits = combined @ self.affine_W + self.affine_b          # (N, V)
            pred = logits.argmax(axis=1)                               # (N,)
            sampled.append(int(pred[0]))
            if int(pred[0]) == eos_id:
                break
            inp = pred[:, np.newaxis]

        return sampled


# ---------------------------------------------------------------------------
# Seq2Seq model
# ---------------------------------------------------------------------------

class Seq2SeqTranslation:
    def __init__(self, ja_vocab_size, en_vocab_size,
                 embed_dim=32, hidden_size=64):
        self.encoder = Encoder(ja_vocab_size, embed_dim, hidden_size)
        self.decoder = AttentionDecoder(en_vocab_size, embed_dim, hidden_size)
        self.params  = self.encoder.params + self.decoder.params
        self.grads   = self.encoder.grads  + self.decoder.grads

    def forward(self, xs, ys_in, ys_out):
        """
        xs     : (N, T_enc) encoder input (Japanese char ids)
        ys_in  : (N, T_dec) decoder input  = [BOS, w1, w2, ..., wT]
        ys_out : (N, T_dec) decoder target = [w1, w2, ..., wT, EOS]
        Returns scalar cross-entropy loss.
        """
        hs_enc = self.encoder.forward(xs)
        h = self.encoder.lstm.h
        c = self.encoder.lstm.c
        self.decoder.set_state(h, c)

        logits = self.decoder.forward(ys_in, hs_enc)   # (N, T_dec, V)

        N, T, V = logits.shape
        logits_2d   = logits.reshape(N * T, V)
        targets_1d  = ys_out.reshape(N * T)
        probs       = _softmax(logits_2d)
        loss        = -np.log(probs[np.arange(N * T), targets_1d] + 1e-7).mean()

        self._cache = (probs, targets_1d, N, T, V, hs_enc)
        return loss

    def backward(self):
        probs, targets_1d, N, T, V, hs_enc = self._cache

        dlogits_2d = probs.copy()
        dlogits_2d[np.arange(N * T), targets_1d] -= 1
        dlogits_2d /= (N * T)

        dlogits = dlogits_2d.reshape(N, T, V)
        dhs_enc = self.decoder.backward(dlogits)
        self.encoder.backward(dhs_enc)

    def translate(self, xs, bos_id, eos_id, max_len=20):
        """Translate a single source sequence (xs: (1, T_enc))."""
        hs_enc = self.encoder.forward(xs)
        h = self.encoder.lstm.h
        c = self.encoder.lstm.c
        self.decoder.lstm.set_state(h, c)
        return self.decoder.generate(hs_enc, bos_id, max_len, eos_id)


# ---------------------------------------------------------------------------
# BLEU-1
# ---------------------------------------------------------------------------

def bleu1(references, hypotheses):
    """Unigram precision averaged over sentence pairs."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        if len(hyp) == 0:
            scores.append(0.0)
            continue
        ref_counts = {}
        for w in ref:
            ref_counts[w] = ref_counts.get(w, 0) + 1
        clip_count = 0
        for w in hyp:
            if ref_counts.get(w, 0) > 0:
                clip_count += 1
                ref_counts[w] -= 1
        scores.append(clip_count / len(hyp))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    t0 = time.time()

    # Build data
    pairs = build_dataset()

    ja_w2i, ja_i2w = build_ja_vocab(pairs)
    en_w2i, en_i2w = build_en_vocab(pairs)
    ja_vocab_size = len(ja_w2i)
    en_vocab_size = len(en_w2i)

    print(f"Dataset: {len(pairs)} pairs")
    print(f"JA vocab (char-level): {ja_vocab_size}")
    print(f"EN vocab (word-level): {en_vocab_size}")
    print(f"Example: '{pairs[0][0]}' → '{pairs[0][1]}'")

    BOS_ID = en_w2i[BOS]
    EOS_ID = en_w2i[EOS]
    PAD_ID = en_w2i[PAD_EN]

    # Encode
    ja_seqs = [encode_ja(ja, ja_w2i) for ja, _ in pairs]
    en_seqs = [encode_en(en, en_w2i) for _, en in pairs]

    ja_padded = pad_sequences(ja_seqs, pad_id=0)
    en_padded  = pad_sequences(en_seqs, pad_id=PAD_ID)
    # ys_in: [BOS, w1, ..., wT], ys_out: [w1, ..., wT, EOS]
    ys_in  = en_padded[:, :-1]
    ys_out = en_padded[:, 1:]

    # Train / test split
    n_test  = 20
    n_train = len(pairs) - n_test
    x_train, x_test  = ja_padded[:n_train],  ja_padded[n_train:]
    yi_train, yi_test = ys_in[:n_train],      ys_in[n_train:]
    yo_train, yo_test = ys_out[:n_train],     ys_out[n_train:]

    # Reference English sentences for BLEU
    test_ref = [en.split() for _, en in pairs[n_train:]]

    # Model
    model = Seq2SeqTranslation(
        ja_vocab_size=ja_vocab_size,
        en_vocab_size=en_vocab_size,
        embed_dim=32,
        hidden_size=64,
    )
    optimizer = Adam(lr=1e-3)

    # Training
    batch_size = 32
    n_train_data = x_train.shape[0]
    print(f"\n--- Training (300 epochs, batch={batch_size}) ---")

    for epoch in range(1, 301):
        idx = np.random.permutation(n_train_data)
        x_sh  = x_train[idx]
        yi_sh = yi_train[idx]
        yo_sh = yo_train[idx]
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train_data, batch_size):
            # Reset encoder LSTM state before each batch
            model.encoder.lstm.reset_state()
            model.decoder.lstm.reset_state()

            xb  = x_sh[start:start + batch_size]
            yib = yi_sh[start:start + batch_size]
            yob = yo_sh[start:start + batch_size]

            loss = model.forward(xb, yib, yob)
            model.backward()
            clip_grads(model.grads, 5.0)
            optimizer.update(model.params, model.grads)
            total_loss += loss
            n_batches  += 1

        if epoch % 50 == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch:3d}  loss = {avg:.4f}")

    elapsed = time.time() - t0
    print(f"\nRuntime: {elapsed:.1f}s")

    # Translate test set for BLEU
    hypotheses = []
    for i in range(n_test):
        model.encoder.lstm.reset_state()
        model.decoder.lstm.reset_state()
        xs = x_test[i:i+1]   # (1, T_enc)
        pred_ids = model.translate(xs, BOS_ID, EOS_ID, max_len=20)
        # Remove EOS and PAD
        pred_words = []
        for pid in pred_ids:
            w = en_i2w.get(pid, "")
            if w in (EOS, PAD_EN, ""):
                break
            pred_words.append(w)
        hypotheses.append(pred_words)

    bleu = bleu1(test_ref, hypotheses)
    print(f"\nBLEU-1 on {n_test} test pairs: {bleu*100:.1f}%")

    # Demo: translate 10 sentences
    demo_pairs = list(zip(pairs[n_train:n_train+10],
                          test_ref[:10], hypotheses[:10]))

    print("\n--- Translation Demo (10 sentences) ---")
    results_demo = []
    for (ja_sent, en_sent), ref_words, hyp_words in demo_pairs:
        pred_str = " ".join(hyp_words)
        ref_str  = " ".join(ref_words)
        match    = pred_str == ref_str
        print(f"  JA : {ja_sent}")
        print(f"  REF: {ref_str}")
        print(f"  HYP: {pred_str}  {'✓' if match else '✗'}")
        print()
        results_demo.append({
            "japanese": ja_sent,
            "reference": ref_str,
            "hypothesis": pred_str,
            "match": match,
        })

    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "experiments", "02-seq2seq-translation")
    os.makedirs(out_dir, exist_ok=True)
    results = {
        "bleu1": round(bleu * 100, 2),
        "n_test": n_test,
        "runtime_sec": round(elapsed, 1),
        "ja_vocab_size": ja_vocab_size,
        "en_vocab_size": en_vocab_size,
        "demo": results_demo,
    }
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
