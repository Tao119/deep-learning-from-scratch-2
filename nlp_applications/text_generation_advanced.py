"""
text_generation_advanced.py

Advanced Text Generation Techniques — pure NumPy LSTM language model.

Sampling strategies:
  1. Greedy decoding (argmax)
  2. Temperature sampling (temp = 0.5, 0.8, 1.0, 1.5)
  3. Top-k sampling (k = 5, 10, 20)
  4. Nucleus (top-p) sampling (p = 0.9, 0.95)

Corpus: 100 Japanese sentences about animals and nature.
Results saved to: experiments/03-text-generation/results.json
"""

import sys
import os
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.time_layers import LSTM, _sigmoid

np.random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "experiments", "03-text-generation")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

CORPUS_SENTENCES = [
    "猫は木の上で眠る。",
    "犬は川のそばで遊ぶ。",
    "鳥は空高く飛ぶ。",
    "魚は川の中を泳ぐ。",
    "花は春に美しく咲く。",
    "木は山の中に静かに立つ。",
    "山は青く美しい。",
    "川は静かに流れる。",
    "空は青くて広い。",
    "海は深くて青い。",
    "猫はよく昼寝をする動物だ。",
    "犬は忠実な動物として知られる。",
    "鳥は朝早く美しい声で鳴く。",
    "魚は群れをなして泳ぐことが多い。",
    "花の香りは心を落ち着かせる。",
    "大きな木の下で休むのは気持ちよい。",
    "山の頂上からの景色は素晴らしい。",
    "川のせせらぎは眠りを誘う。",
    "雲が空をゆっくりと流れていく。",
    "波が海岸に打ち寄せる音は美しい。",
    "猫は夜行性の動物で夜に活発になる。",
    "犬は人間と長い歴史を持つ動物だ。",
    "渡り鳥は遠い国へと旅をする。",
    "川魚は清流に住む美しい生き物だ。",
    "桜の花は春の象徴として愛される。",
    "森の木々は風に揺れてざわめく。",
    "高い山には雪が積もる。",
    "清流は山の湧き水から生まれる。",
    "夕焼け空は橙色に染まる。",
    "満潮のとき海は力強く岸を打つ。",
    "猫は水が苦手な動物が多い。",
    "犬は嗅覚が非常に優れている。",
    "小鳥は草の実を好んで食べる。",
    "川エビは石の下に隠れて暮らす。",
    "梅の花は寒い冬の終わりに咲く。",
    "竹林の中は涼しくて静かだ。",
    "富士山は日本一高い山として有名だ。",
    "利根川は関東平野を流れる大きな川だ。",
    "夜空には無数の星が輝く。",
    "南の海には色鮮やかな魚が泳ぐ。",
    "猫は高い場所が好きな動物だ。",
    "犬は走ることが大好きだ。",
    "ツバメは毎年春に日本に戻ってくる。",
    "鯉は長生きする魚として知られる。",
    "チューリップは春の公園を彩る。",
    "松の木は厳しい冬にも青々としている。",
    "アルプスの山々は壮大な景観を見せる。",
    "アマゾン川は世界最大の流域を持つ。",
    "北の空にオーロラが輝く夜がある。",
    "珊瑚礁は海の生き物の楽園だ。",
    "猫のひげは方向感覚を助ける役割がある。",
    "犬の耳は小さな音も聞き取れる。",
    "鷹は高い空から獲物を狙う。",
    "鮭は川を遡って産卵する。",
    "向日葵は太陽に向かって咲く花だ。",
    "椿の花は冬にも鮮やかに咲く。",
    "エベレストは世界で最も高い山だ。",
    "ナイル川はアフリカ最長の川だ。",
    "星空の下でキャンプをするのは楽しい。",
    "深海には未知の生き物が多く住む。",
    "猫は独立心の強い動物として知られる。",
    "犬は集団行動を好む社会的な動物だ。",
    "コウモリは夜に超音波で飛ぶ。",
    "ウナギは川と海を行き来して生きる。",
    "菜の花は春の野原を黄色に染める。",
    "ケヤキは秋に美しく紅葉する。",
    "日本アルプスは雄大な自然を誇る。",
    "最上川は山形県を流れる清流だ。",
    "三日月が夜空にかかる夜は幻想的だ。",
    "クジラは海の中で美しい歌を歌う。",
    "猫は清潔好きで毎日毛づくろいをする。",
    "犬は人を助ける盲導犬として活躍する。",
    "フクロウは夜に目が利く鳥だ。",
    "イカは海の中で素早く動く。",
    "コスモスは秋風に揺れる美しい花だ。",
    "紅葉した森を歩くのは心が弾む。",
    "雪山は冬の厳しさと美しさを示す。",
    "川の流れは時間の流れに例えられる。",
    "月明かりの下で夜景を楽しむ。",
    "深い海の底には光が届かない。",
    "猫は狭い場所に入ることが好きだ。",
    "犬は主人の感情を敏感に感じ取る。",
    "カモメは海辺を自由に飛び回る。",
    "タコは知性的な海の生き物だ。",
    "芙蓉の花は夏に大きく咲く。",
    "クヌギの木はドングリの実をつける。",
    "山道を歩くと自然の力を感じる。",
    "清流で釣りをするのは至福の時間だ。",
    "夜明けの空はピンクと紫が混じる。",
    "珊瑚は海の生態系を支える存在だ。",
    "猫は狩りの本能を持っている。",
    "犬は優れた嗅覚で犯罪捜査に使われる。",
    "キツツキは木をつついて虫を捕る。",
    "ヒラメは砂底に隠れる魚だ。",
    "ポピーは風に揺れる可憐な花だ。",
    "サクラの木は古くから日本人に愛される。",
    "山の中腹に咲く高山植物は美しい。",
    "大河の流れは悠然として止まらない。",
    "星座を眺めながら宇宙の広さを感じる。",
    "海の深さは人間の知識をはるかに超える。",
]


# ---------------------------------------------------------------------------
# Vocabulary (character-level)
# ---------------------------------------------------------------------------

def build_vocab(corpus):
    chars = set()
    for sent in corpus:
        chars.update(sent)
    vocab = ["<PAD>", "<BOS>", "<EOS>"] + sorted(chars)
    ch2id = {c: i for i, c in enumerate(vocab)}
    id2ch = {i: c for c, i in ch2id.items()}
    return vocab, ch2id, id2ch


def encode_corpus(sentences, ch2id):
    bos = ch2id["<BOS>"]
    eos = ch2id["<EOS>"]
    sequences = []
    for sent in sentences:
        seq = [bos] + [ch2id.get(c, 0) for c in sent] + [eos]
        sequences.append(seq)
    return sequences


def make_training_pairs(sequences, seq_len=20):
    xs, ys = [], []
    for seq in sequences:
        for start in range(0, len(seq) - 1, seq_len):
            x_chunk = seq[start:start + seq_len]
            y_chunk = seq[start + 1:start + seq_len + 1]
            if len(x_chunk) < 2:
                continue
            xs.append(x_chunk)
            ys.append(y_chunk)
    return xs, ys


def pad_batch(xs, ys, pad_id=0):
    max_len = max(len(x) for x in xs)
    xb = np.full((len(xs), max_len), pad_id, dtype=np.int32)
    yb = np.full((len(ys), max_len), -1, dtype=np.int32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xb[i, :len(x)] = x
        yb[i, :len(y)] = y
    return xb, yb


# ---------------------------------------------------------------------------
# Embedding layer
# ---------------------------------------------------------------------------

class Embedding:
    def __init__(self, V, D):
        self.W = (np.random.randn(V, D) * 0.01).astype("f")
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
# Linear layer
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
# LSTM Language Model
# ---------------------------------------------------------------------------

class LSTMLanguageModel:
    def __init__(self, vocab_size, emb_dim, hidden_size):
        self.embedding = Embedding(vocab_size, emb_dim)
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size

        scale = np.sqrt(emb_dim + hidden_size)
        Wx = (np.random.randn(emb_dim, 4 * hidden_size) / scale).astype("f")
        Wh = (np.random.randn(hidden_size, 4 * hidden_size) / scale).astype("f")
        b  = np.zeros(4 * hidden_size, dtype="f")
        self._lstm_cell = LSTM(Wx, Wh, b)
        self._lstm_params = self._lstm_cell.params
        self._lstm_grads  = self._lstm_cell.grads

        self.linear = Linear(hidden_size, vocab_size)

        self.params = (self.embedding.params
                       + self._lstm_params
                       + self.linear.params)
        self.grads  = (self.embedding.grads
                       + self._lstm_grads
                       + self.linear.grads)
        self._cache = None

    def forward(self, xs, ys):
        emb = self.embedding.forward(xs)
        N, T, D = emb.shape
        H = self.hidden_size

        h = np.zeros((N, H), dtype="f")
        c = np.zeros((N, H), dtype="f")
        hs = []
        cell_caches = []
        for t in range(T):
            h, c = self._lstm_cell.forward(emb[:, t, :], h, c)
            hs.append(h)
            cell_caches.append(self._lstm_cell.cache)

        hs_arr = np.stack(hs, axis=1)
        logits = self.linear.forward(hs_arr)

        flat_logits = logits.reshape(N * T, self.vocab_size)
        flat_labels = ys.reshape(N * T)
        mask = flat_labels >= 0
        ml = flat_logits[mask]
        tl = flat_labels[mask]

        if len(tl) == 0:
            self._cache = None
            return 0.0

        ml_max = ml.max(axis=1, keepdims=True)
        exp_ml = np.exp(ml - ml_max)
        probs  = exp_ml / exp_ml.sum(axis=1, keepdims=True)
        loss   = -np.log(probs[np.arange(len(tl)), tl] + 1e-7).mean()

        self._cache = (emb, ys, probs, tl, mask, N, T, hs_arr, cell_caches, h, c)
        return loss

    def backward(self):
        if self._cache is None:
            return
        emb, ys, probs, tl, mask, N, T, hs_arr, cell_caches, _, _ = self._cache

        d_probs = probs.copy()
        d_probs[np.arange(len(tl)), tl] -= 1
        d_probs /= len(tl)

        flat_labels = ys.reshape(N * T)
        d_flat = np.zeros((N * T, self.vocab_size), dtype="f")
        d_flat[mask] = d_probs
        d_logits = d_flat.reshape(N, T, self.vocab_size)

        d_hs = self.linear.backward(d_logits)

        dh = np.zeros((N, self.hidden_size), dtype="f")
        dc = np.zeros((N, self.hidden_size), dtype="f")
        dWx_total = np.zeros_like(self._lstm_params[0])
        dWh_total = np.zeros_like(self._lstm_params[1])
        db_total  = np.zeros_like(self._lstm_params[2])
        demb = np.zeros_like(emb)

        for t in reversed(range(T)):
            self._lstm_cell.cache = cell_caches[t]
            dx_t, dh, dc = self._lstm_cell.backward(d_hs[:, t, :] + dh, dc)
            demb[:, t, :] = dx_t
            dWx_total += self._lstm_grads[0]
            dWh_total += self._lstm_grads[1]
            db_total  += self._lstm_grads[2]

        self._lstm_grads[0][...] = dWx_total
        self._lstm_grads[1][...] = dWh_total
        self._lstm_grads[2][...] = db_total

        self.embedding.backward(demb)

    def predict_next(self, token_id, h, c):
        """Single-step prediction; returns (logits, h_next, c_next)."""
        x = np.array([[token_id]], dtype=np.int32)
        emb = self.embedding.forward(x)
        h_next, c_next = self._lstm_cell.forward(emb[:, 0, :], h, c)
        self._lstm_cell.cache  # keep cache valid for potential backward
        logits = (h_next @ self.linear.W + self.linear.b)[0]
        return logits, h_next, c_next


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def greedy_decode(model, ch2id, id2ch, seed_text, max_len=40):
    h = np.zeros((1, model.hidden_size), dtype="f")
    c = np.zeros((1, model.hidden_size), dtype="f")
    tokens = [ch2id["<BOS>"]] + [ch2id.get(c, 0) for c in seed_text]
    result = seed_text
    for tok in tokens[:-1]:
        logits, h, c = model.predict_next(tok, h, c)
    cur = tokens[-1]
    for _ in range(max_len):
        logits, h, c = model.predict_next(cur, h, c)
        cur = int(np.argmax(logits))
        if cur == ch2id["<EOS>"]:
            break
        result += id2ch.get(cur, "?")
    return result


def temperature_decode(model, ch2id, id2ch, seed_text, temp=1.0, max_len=40):
    h = np.zeros((1, model.hidden_size), dtype="f")
    c = np.zeros((1, model.hidden_size), dtype="f")
    tokens = [ch2id["<BOS>"]] + [ch2id.get(ch, 0) for ch in seed_text]
    result = seed_text
    for tok in tokens[:-1]:
        logits, h, c = model.predict_next(tok, h, c)
    cur = tokens[-1]
    for _ in range(max_len):
        logits, h, c = model.predict_next(cur, h, c)
        probs = _softmax(logits / (temp + 1e-8))
        cur = int(np.random.choice(len(probs), p=probs))
        if cur == ch2id["<EOS>"]:
            break
        result += id2ch.get(cur, "?")
    return result


def topk_decode(model, ch2id, id2ch, seed_text, k=10, max_len=40):
    h = np.zeros((1, model.hidden_size), dtype="f")
    c = np.zeros((1, model.hidden_size), dtype="f")
    tokens = [ch2id["<BOS>"]] + [ch2id.get(ch, 0) for ch in seed_text]
    result = seed_text
    for tok in tokens[:-1]:
        logits, h, c = model.predict_next(tok, h, c)
    cur = tokens[-1]
    for _ in range(max_len):
        logits, h, c = model.predict_next(cur, h, c)
        top_k_idx = np.argsort(logits)[-k:]
        top_k_log = logits[top_k_idx]
        probs = _softmax(top_k_log)
        cur = int(top_k_idx[np.random.choice(k, p=probs)])
        if cur == ch2id["<EOS>"]:
            break
        result += id2ch.get(cur, "?")
    return result


def nucleus_decode(model, ch2id, id2ch, seed_text, p=0.9, max_len=40):
    h = np.zeros((1, model.hidden_size), dtype="f")
    c = np.zeros((1, model.hidden_size), dtype="f")
    tokens = [ch2id["<BOS>"]] + [ch2id.get(ch, 0) for ch in seed_text]
    result = seed_text
    for tok in tokens[:-1]:
        logits, h, c = model.predict_next(tok, h, c)
    cur = tokens[-1]
    for _ in range(max_len):
        logits, h, c = model.predict_next(cur, h, c)
        sorted_idx = np.argsort(logits)[::-1]
        sorted_probs = _softmax(logits[sorted_idx])
        cumprobs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumprobs, p) + 1
        nucleus_idx = sorted_idx[:cutoff]
        nucleus_probs = _softmax(logits[nucleus_idx])
        cur = int(nucleus_idx[np.random.choice(len(nucleus_idx), p=nucleus_probs)])
        if cur == ch2id["<EOS>"]:
            break
        result += id2ch.get(cur, "?")
    return result


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def diversity_metrics(texts):
    all_chars = []
    for t in texts:
        all_chars.extend(list(t))
    if not all_chars:
        return {"unique_ratio": 0.0, "repetition_rate": 0.0}
    unique_ratio = len(set(all_chars)) / len(all_chars)
    # Repetition rate: fraction of bigrams that repeat
    bigrams = [(all_chars[i], all_chars[i + 1]) for i in range(len(all_chars) - 1)]
    if not bigrams:
        return {"unique_ratio": unique_ratio, "repetition_rate": 0.0}
    rep_rate = 1 - len(set(bigrams)) / len(bigrams)
    return {"unique_ratio": round(unique_ratio, 4), "repetition_rate": round(rep_rate, 4)}


# ---------------------------------------------------------------------------
# Adam
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
# Training + generation
# ---------------------------------------------------------------------------

def train_lm(corpus, n_epochs=80, batch_size=32, seq_len=20):
    vocab, ch2id, id2ch = build_vocab(corpus)
    sequences = encode_corpus(corpus, ch2id)
    raw_xs, raw_ys = make_training_pairs(sequences, seq_len=seq_len)

    vocab_size  = len(vocab)
    emb_dim     = 32
    hidden_size = 64
    model       = LSTMLanguageModel(vocab_size, emb_dim, hidden_size)
    optimizer   = Adam(lr=1e-3)

    print("=" * 55)
    print("LSTM Language Model Training")
    print(f"  corpus={len(corpus)} sentences  vocab={vocab_size}")
    print("=" * 55)

    n_samples = len(raw_xs)
    for epoch in range(1, n_epochs + 1):
        idx = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, n_samples, batch_size):
            b_idx = idx[start:start + batch_size]
            batch_xs = [raw_xs[i] for i in b_idx]
            batch_ys = [raw_ys[i] for i in b_idx]
            xb, yb = pad_batch(batch_xs, batch_ys)
            loss = model.forward(xb, yb)
            model.backward()
            total_norm = np.sqrt(sum((g ** 2).sum() for g in model.grads))
            if total_norm > 5.0:
                for g in model.grads:
                    g *= 5.0 / (total_norm + 1e-8)
            optimizer.update(model.params, model.grads)
            total_loss += loss
            n_batches  += 1
        if epoch % 20 == 0:
            print(f"epoch {epoch:>3}  loss={total_loss / n_batches:.4f}")

    return model, ch2id, id2ch


if __name__ == "__main__":
    model, ch2id, id2ch = train_lm(CORPUS_SENTENCES, n_epochs=80, batch_size=32)

    seed = "猫は"
    N_SAMPLES = 5
    results = {}

    # 1. Greedy
    print("\n--- Greedy Decoding ---")
    greedy_texts = [greedy_decode(model, ch2id, id2ch, seed) for _ in range(N_SAMPLES)]
    for t in greedy_texts:
        print(f"  {t}")
    metrics = diversity_metrics(greedy_texts)
    results["greedy"] = {"texts": greedy_texts, "metrics": metrics}
    print(f"  diversity: {metrics}")

    # 2. Temperature sampling
    print("\n--- Temperature Sampling ---")
    results["temperature"] = {}
    for temp in [0.5, 0.8, 1.0, 1.5]:
        texts = [temperature_decode(model, ch2id, id2ch, seed, temp=temp)
                 for _ in range(N_SAMPLES)]
        metrics = diversity_metrics(texts)
        results["temperature"][f"temp_{temp}"] = {"texts": texts, "metrics": metrics}
        print(f"  temp={temp}: {texts[0]}")
        print(f"    diversity: {metrics}")

    # 3. Top-k sampling
    print("\n--- Top-k Sampling ---")
    results["topk"] = {}
    for k in [5, 10, 20]:
        texts = [topk_decode(model, ch2id, id2ch, seed, k=k) for _ in range(N_SAMPLES)]
        metrics = diversity_metrics(texts)
        results["topk"][f"k_{k}"] = {"texts": texts, "metrics": metrics}
        print(f"  k={k}: {texts[0]}")
        print(f"    diversity: {metrics}")

    # 4. Nucleus (top-p) sampling
    print("\n--- Nucleus (Top-p) Sampling ---")
    results["nucleus"] = {}
    for p in [0.9, 0.95]:
        texts = [nucleus_decode(model, ch2id, id2ch, seed, p=p) for _ in range(N_SAMPLES)]
        metrics = diversity_metrics(texts)
        results["nucleus"][f"p_{p}"] = {"texts": texts, "metrics": metrics}
        print(f"  p={p}: {texts[0]}")
        print(f"    diversity: {metrics}")

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
