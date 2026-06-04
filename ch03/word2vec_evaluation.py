"""
ch03/word2vec_evaluation.py — Word2Vec Evaluation Suite

Comprehensive evaluation of three word embedding models:
  1. CBOW (Continuous Bag-of-Words)
  2. Skip-gram
  3. Subword (character n-gram augmented CBOW)

Evaluation axes:
  Intrinsic 1 — Word analogy accuracy (Japanese analogies: capital:country)
  Intrinsic 2 — Word similarity correlation (human ratings vs cosine sim)
  Extrinsic   — Downstream sentiment classification (BoE + logistic regression)

All models trained on the same synthetic Japanese corpus (~500 sentences).
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.embedding import NegativeSamplingLoss, Embedding
from common.optimizer import Adam

OUT_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)


# ===========================================================================
# Corpus (reused from word2vec_analogy.py, expanded for better coverage)
# ===========================================================================

CORPUS_SENTENCES = [
    # Geography / capitals
    "東京 は 日本 の 首都 です",
    "パリ は フランス の 首都 です",
    "ロンドン は イギリス の 首都 です",
    "ベルリン は ドイツ の 首都 です",
    "北京 は 中国 の 首都 です",
    "ソウル は 韓国 の 首都 です",
    "ワシントン は アメリカ の 首都 です",
    "ローマ は イタリア の 首都 です",
    "日本 の 首都 は 東京 です",
    "フランス の 首都 は パリ です",
    "中国 の 首都 は 北京 です",
    "東京 は 大都市 です",
    "パリ は 大都市 です",
    "東京 に は 多く の 人 が 住ん で いる",
    "パリ に は 美術館 が 多い",
    "東京 は アジア の 中心 です",
    "日本 は アジア に ある 国 です",
    "フランス は ヨーロッパ に ある 国 です",
    "ドイツ は ヨーロッパ に ある 国 です",
    "アメリカ は 大きな 国 です",
    # Animals / pets
    "犬 は ペット です",
    "猫 は ペット です",
    "犬 は 人 に 懐く 動物 です",
    "猫 は 独立心 が 強い 動物 です",
    "犬 は 散歩 が 好き です",
    "猫 は 魚 を 食べ ます",
    "犬 は 骨 が 好き です",
    "ライオン は 野生 の 動物 です",
    "虎 は 野生 の 動物 です",
    "象 は 大きな 動物 です",
    "犬 は 吠える 動物 です",
    "猫 は 鳴く 動物 です",
    "犬 は 忠実 な 動物 です",
    # Professions
    "医者 は 病院 で 働く",
    "先生 は 学校 で 働く",
    "料理人 は レストラン で 働く",
    "警察官 は 交番 で 働く",
    "消防士 は 消防署 で 働く",
    "農家 は 農場 で 働く",
    "弁護士 は 法律事務所 で 働く",
    "エンジニア は 会社 で 働く",
    "研究者 は 大学 で 働く",
    "医者 は 患者 を 診る",
    "先生 は 生徒 を 教える",
    # Food
    "寿司 は 日本 の 食べ物 です",
    "ラーメン は 人気 の 食べ物 です",
    "てんぷら は 日本 料理 です",
    "刺身 は 新鮮 な 魚 を 使う",
    "味噌汁 は 毎朝 飲む",
    "寿司 は 魚 と ご飯 で 作る",
    # Transport
    "新幹線 は 速い 電車 です",
    "飛行機 は 空 を 飛ぶ",
    "電車 は 駅 に 止まる",
    "バス は 道路 を 走る",
    "自動車 は 道路 を 走る",
    "新幹線 は 東京 と 大阪 を 結ぶ",
    # Sentiment-related (positive)
    "この 映画 は 素晴らしい です",
    "料理 が 美味しい",
    "景色 が 美しい",
    "サービス が 良い",
    "楽しい 旅行 でした",
    "感動 的 な 体験 でした",
    "最高 の 一日 でした",
    "素敵 な 場所 です",
    "笑顔 で 帰れ ます",
    "また 来 たい です",
    # Sentiment-related (negative)
    "この サービス は 最悪 です",
    "料理 が まずい",
    "待ち時間 が 長い",
    "対応 が 悪い",
    "残念 な 体験 でした",
    "がっかり し ました",
    "二度 と 来 ない",
    "品質 が 低い",
    "問題 が 多い",
    "不満 が あります",
    # Animals continued
    "犬 は 人間 の 友達 です",
    "猫 は 賢い 動物 です",
    "犬 と 猫 は どちら も ペット です",
    "病院 と 学校 は 大切 な 場所 です",
    "日本 は アジア で 有名 な 国 です",
    "新幹線 は 速く て 便利 な 電車 です",
    "飛行機 は 速い 乗り物 です",
    # Expanded geography
    "東京 は 大きな 首都 です",
    "パリ は 有名 な 首都 です",
    "北京 は 中国 の 大都市 です",
    "ベルリン は ドイツ の 首都 で す",
    "ソウル は 韓国 の 大都市 です",
    # Extra positive
    "この 商品 は 素晴らしい",
    "品質 が 高い",
    "満足 しています",
    "優秀 な スタッフ でした",
    "感謝 しています",
    # Extra negative
    "この 映画 は つまらない",
    "期待 外れ でした",
    "高い の に 価値 が ない",
    "時間 を 無駄 に した",
    "おすすめ でき ない",
]
# Expand to ~500 by repeating
_EXT = [
    "日本 は 美しい 国 です",
    "フランス は 文化 の 国 です",
    "医者 は 病院 に いる",
    "先生 は 学校 に いる",
    "電車 は 便利 です",
    "犬 は 公園 で 遊ぶ",
    "猫 は 家 で 過ごす",
    "この 体験 は 最高 でした",
    "サービス が 最悪 でした",
]
CORPUS_SENTENCES = (CORPUS_SENTENCES + _EXT * 12)[:500]


# ===========================================================================
# Vocabulary & corpus helpers
# ===========================================================================

def build_vocab_corpus(sentences: list[str]) -> tuple[np.ndarray, dict, dict]:
    word_to_id: dict[str, int] = {}
    id_to_word: dict[int, str] = {}
    corpus: list[int] = []
    for sent in sentences:
        for word in sent.split():
            if word not in word_to_id:
                wid = len(word_to_id)
                word_to_id[word] = wid
                id_to_word[wid] = word
            corpus.append(word_to_id[word])
    return np.array(corpus, dtype=np.int32), word_to_id, id_to_word


def create_contexts_target(corpus: np.ndarray, window: int = 2
                           ) -> tuple[np.ndarray, np.ndarray]:
    targets = corpus[window:-window]
    contexts = []
    for i in range(window, len(corpus) - window):
        ctx = ([corpus[i - w] for w in range(window, 0, -1)]
               + [corpus[i + w] for w in range(1, window + 1)])
        contexts.append(ctx)
    return np.array(contexts, dtype=np.int32), np.array(targets, dtype=np.int32)


def cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


# ===========================================================================
# Embedding layer (simple, no library)
# ===========================================================================

class _Embedding:
    """Minimal embedding lookup layer for subword model."""
    def __init__(self, W: np.ndarray):
        self.params = [W]
        self.grads  = [np.zeros_like(W)]
        self._idx: np.ndarray | None = None

    def forward(self, idx: np.ndarray) -> np.ndarray:
        self._idx = idx
        return self.params[0][idx]

    def backward(self, dout: np.ndarray) -> None:
        dW = self.grads[0]
        dW[...] = 0
        np.add.at(dW, self._idx, dout)


# ===========================================================================
# Model 1: CBOW
# ===========================================================================

class CBOWModel:
    def __init__(self, vocab_size: int, hidden_size: int, corpus: np.ndarray, window: int = 2):
        V, H = vocab_size, hidden_size
        W_in  = (np.random.randn(V, H) / 100).astype(np.float32)
        W_out = (np.random.randn(V, H) / 100).astype(np.float32)

        self.in_layers = [_Embedding(W_in) for _ in range(2 * window)]
        self.ns_loss   = NegativeSamplingLoss(W_out, corpus, sample_size=5)

        self.params = []
        self.grads  = []
        for layer in self.in_layers:
            self.params += layer.params
            self.grads  += layer.grads
        self.params += self.ns_loss.params
        self.grads  += self.ns_loss.grads

        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        h = np.zeros((contexts.shape[0], self.in_layers[0].params[0].shape[1]), dtype=np.float32)
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h /= len(self.in_layers)
        return self.ns_loss.forward(h, target)

    def backward(self) -> None:
        dout = self.ns_loss.backward()
        dout /= len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)


# ===========================================================================
# Model 2: Skip-gram
# ===========================================================================

class SkipGramModel:
    def __init__(self, vocab_size: int, hidden_size: int, corpus: np.ndarray, window: int = 2):
        V, H = vocab_size, hidden_size
        W_in  = (0.01 * np.random.randn(V, H)).astype(np.float32)
        W_out = (0.01 * np.random.randn(V, H)).astype(np.float32)

        self.in_layer  = Embedding(W_in)
        self.loss_layers = [
            NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
            for _ in range(2 * window)
        ]

        all_layers = [self.in_layer] + self.loss_layers
        self.params, self.grads = [], []
        for l in all_layers:
            self.params += l.params
            self.grads  += l.grads

        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        h = self.in_layer.forward(target)
        loss: float = 0.0
        for i, layer in enumerate(self.loss_layers):
            loss += layer.forward(h, contexts[:, i])
        return loss

    def backward(self) -> None:
        dh_sum = None
        for layer in self.loss_layers:
            dh = layer.backward()
            if dh_sum is None:
                dh_sum = dh.copy()
            else:
                dh_sum = dh_sum + dh
        if dh_sum is not None:
            self.in_layer.backward(dh_sum)


# ===========================================================================
# Model 3: Subword CBOW (character n-gram augmented)
# ===========================================================================

def _word_to_subwords(word: str, min_n: int = 2, max_n: int = 3) -> list[str]:
    """Decompose a Japanese word into character n-grams."""
    chars = list(word)
    ngrams: list[str] = [word]  # include word itself
    for n in range(min_n, max_n + 1):
        for i in range(len(chars) - n + 1):
            ngrams.append("".join(chars[i:i+n]))
    return list(set(ngrams))


def build_subword_vocab(word_to_id: dict[str, int]) -> dict[str, int]:
    """Build extended vocabulary that includes subword tokens."""
    sub_vocab = dict(word_to_id)  # start with word vocab
    for word in word_to_id:
        for sub in _word_to_subwords(word):
            if sub not in sub_vocab:
                sub_vocab[sub] = len(sub_vocab)
    return sub_vocab


def word_to_subword_vector(word: str, sub_vocab: dict[str, int],
                            sub_emb: np.ndarray) -> np.ndarray:
    """Average subword embeddings for a word."""
    subs = _word_to_subwords(word)
    vecs = [sub_emb[sub_vocab[s]] for s in subs if s in sub_vocab]
    if not vecs:
        return np.zeros(sub_emb.shape[1], dtype=np.float32)
    return np.mean(vecs, axis=0)


class SubwordCBOW:
    """CBOW where each word's embedding = average of its subword embeddings."""

    def __init__(self, word_to_id: dict[str, int], hidden_size: int,
                 corpus: np.ndarray, window: int = 2):
        self.word_to_id = word_to_id
        self.sub_vocab = build_subword_vocab(word_to_id)
        V_sub = len(self.sub_vocab)
        H = hidden_size

        W_in  = (np.random.randn(V_sub, H) / 100).astype(np.float32)
        W_out = (np.random.randn(V_sub, H) / 100).astype(np.float32)

        self.sub_emb  = W_in
        self.ns_loss  = NegativeSamplingLoss(W_out, corpus, sample_size=5)
        self.params   = [W_in] + self.ns_loss.params
        self.grads    = [np.zeros_like(W_in)] + self.ns_loss.grads
        self.window   = window

    def _lookup(self, word_ids: np.ndarray) -> np.ndarray:
        """(batch, window) → average subword embeddings."""
        id_to_word = {v: k for k, v in self.word_to_id.items()}
        H = self.sub_emb.shape[1]
        batch, win = word_ids.shape
        out = np.zeros((batch, H), dtype=np.float32)
        for b in range(batch):
            for w in range(win):
                wid = int(word_ids[b, w])
                word = id_to_word.get(wid, "")
                out[b] += word_to_subword_vector(word, self.sub_vocab, self.sub_emb)
            out[b] /= win
        return out

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        h = self._lookup(contexts)
        return float(self.ns_loss.forward(h, target))

    def backward(self) -> None:
        self.ns_loss.backward()

    def get_word_vecs(self) -> np.ndarray:
        """Build word vectors from subword embeddings."""
        V = len(self.word_to_id)
        H = self.sub_emb.shape[1]
        vecs = np.zeros((V, H), dtype=np.float32)
        for word, wid in self.word_to_id.items():
            vecs[wid] = word_to_subword_vector(word, self.sub_vocab, self.sub_emb)
        return vecs


# ===========================================================================
# Training
# ===========================================================================

def train_model(model_name: str, model, contexts: np.ndarray, targets: np.ndarray,
                n_epochs: int = 100, batch_size: int = 64, lr: float = 0.001,
                verbose: bool = True) -> list[float]:
    optimizer = Adam(lr=lr)
    N = len(targets)
    losses: list[float] = []
    rng = np.random.default_rng(42)

    for epoch in range(n_epochs):
        idx = rng.permutation(N)
        ctxs, tgts = contexts[idx], targets[idx]
        epoch_loss = 0.0
        n_iter = max(1, N // batch_size)
        for i in range(n_iter):
            c = ctxs[i * batch_size:(i + 1) * batch_size]
            t = tgts[i * batch_size:(i + 1) * batch_size]
            loss = model.forward(c, t)
            model.backward()
            optimizer.update(model.params, model.grads)
            epoch_loss += float(loss)
        mean_loss = epoch_loss / n_iter
        losses.append(mean_loss)
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  [{model_name}] epoch {epoch+1:3d}/{n_epochs}  loss={mean_loss:.4f}")

    return losses


# ===========================================================================
# Intrinsic Eval 1 — Word Analogy (a:b = c:?)
# ===========================================================================

# Japanese analogy pairs: (a, b, c, expected_d)
# Pattern: 国:首都 = 国:首都
ANALOGY_PAIRS = [
    # capital : country
    ("日本", "東京", "フランス", "パリ"),
    ("日本", "東京", "ドイツ",  "ベルリン"),
    ("フランス", "パリ", "日本", "東京"),
    ("フランス", "パリ", "中国", "北京"),
    ("ドイツ",  "ベルリン", "日本", "東京"),
    # pet:animal
    ("犬", "ペット", "猫", "ペット"),
    # workplace
    ("医者", "病院", "先生", "学校"),
    ("先生", "学校", "医者", "病院"),
    ("料理人", "レストラン", "農家", "農場"),
    ("農家", "農場", "料理人", "レストラン"),
]


def eval_analogy(word_vecs: np.ndarray, word_to_id: dict[str, int],
                 id_to_word: dict[int, str]) -> dict:
    correct = 0
    valid = 0
    miss_vocab = []
    details = []

    for (a, b, c, expected) in ANALOGY_PAIRS:
        if any(w not in word_to_id for w in (a, b, c, expected)):
            miss_vocab.append((a, b, c, expected))
            continue
        va = word_vecs[word_to_id[a]]
        vb = word_vecs[word_to_id[b]]
        vc = word_vecs[word_to_id[c]]
        query = vb - va + vc
        norm = np.linalg.norm(query)
        if norm < 1e-8:
            continue
        query /= norm

        # Find nearest neighbour (excluding a, b, c)
        exclude = {word_to_id[w] for w in (a, b, c)}
        sims = np.array([
            cos_sim(word_vecs[i], query) if i not in exclude else -1.0
            for i in range(len(id_to_word))
        ])
        pred_id = int(np.argmax(sims))
        pred_word = id_to_word[pred_id]
        is_correct = pred_word == expected
        if is_correct:
            correct += 1
        valid += 1
        details.append({
            "query": f"{a}:{b}={c}:?",
            "expected": expected,
            "predicted": pred_word,
            "correct": is_correct,
        })

    accuracy = correct / valid if valid > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "valid": valid,
        "missed_vocab": len(miss_vocab),
        "details": details,
    }


# ===========================================================================
# Intrinsic Eval 2 — Word Similarity (Spearman correlation)
# ===========================================================================

# (word1, word2, human_score_0-10)
SIMILARITY_PAIRS: list[tuple[str, str, float]] = [
    ("犬",   "猫",     8.0),  # both pets
    ("犬",   "ペット",  7.0),
    ("猫",   "ペット",  7.0),
    ("医者", "病院",   8.5),  # doctor works at hospital
    ("先生", "学校",   8.5),
    ("東京", "日本",   7.5),  # capital in country
    ("パリ", "フランス", 7.5),
    ("医者", "先生",   4.0),  # both professionals, different domain
    ("犬",   "ライオン", 4.5), # both animals
    ("東京", "パリ",   5.5),  # both capitals
    ("電車", "新幹線",  7.0),
    ("飛行機", "電車", 5.0),
    ("犬",   "飛行機", 0.5),  # unrelated
    ("先生", "農場",   0.5),  # unrelated
    ("寿司", "料理",  7.5),
    ("寿司", "ラーメン", 6.0),
]


def spearman_rank_corr(x: list[float], y: list[float]) -> float:
    """Spearman ρ without scipy."""
    n = len(x)
    if n < 2:
        return 0.0

    def rank(arr):
        sorted_idx = np.argsort(arr)
        ranks = np.zeros(n)
        ranks[sorted_idx] = np.arange(1, n + 1, dtype=float)
        return ranks

    rx = rank(np.array(x))
    ry = rank(np.array(y))
    d2 = np.sum((rx - ry) ** 2)
    return float(1.0 - 6 * d2 / (n * (n ** 2 - 1) + 1e-10))


def eval_similarity(word_vecs: np.ndarray, word_to_id: dict[str, int]) -> dict:
    human_scores: list[float] = []
    model_scores: list[float] = []
    skipped = 0

    for (w1, w2, human) in SIMILARITY_PAIRS:
        if w1 not in word_to_id or w2 not in word_to_id:
            skipped += 1
            continue
        v1 = word_vecs[word_to_id[w1]]
        v2 = word_vecs[word_to_id[w2]]
        human_scores.append(human)
        model_scores.append(cos_sim(v1, v2))

    rho = spearman_rank_corr(human_scores, model_scores) if len(human_scores) >= 4 else 0.0
    return {
        "spearman_rho": round(rho, 4),
        "n_pairs": len(human_scores),
        "n_skipped": skipped,
    }


# ===========================================================================
# Extrinsic Eval — Sentiment Classification (BoE + Logistic Regression)
# ===========================================================================

SENTIMENT_DATA: list[tuple[str, int]] = [
    # Positive (1)
    ("この 映画 は 素晴らしい です", 1),
    ("料理 が 美味しい", 1),
    ("景色 が 美しい", 1),
    ("サービス が 良い", 1),
    ("楽しい 旅行 でした", 1),
    ("感動 的 な 体験 でした", 1),
    ("最高 の 一日 でした", 1),
    ("素敵 な 場所 です", 1),
    ("また 来 たい です", 1),
    ("満足 しています", 1),
    ("品質 が 高い", 1),
    ("優秀 な スタッフ でした", 1),
    ("感謝 しています", 1),
    ("笑顔 で 帰れ ます", 1),
    ("この 商品 は 素晴らしい", 1),
    ("この 体験 は 最高 でした", 1),
    ("居心地 が 良い です", 1),
    ("スタッフ が 親切 です", 1),
    ("価格 に 見合う 品質 です", 1),
    ("大変 満足 です", 1),
    # Negative (0)
    ("このサービス は 最悪 です", 0),
    ("料理 が まずい", 0),
    ("待ち時間 が 長い", 0),
    ("対応 が 悪い", 0),
    ("残念 な 体験 でした", 0),
    ("がっかり し ました", 0),
    ("二度 と 来 ない", 0),
    ("品質 が 低い", 0),
    ("問題 が 多い", 0),
    ("不満 が あります", 0),
    ("この 映画 は つまらない", 0),
    ("期待 外れ でした", 0),
    ("高い の に 価値 が ない", 0),
    ("時間 を 無駄 に した", 0),
    ("おすすめ でき ない", 0),
    ("サービス が 最悪 でした", 0),
    ("不快 な 経験 でした", 0),
    ("クオリティ が 低い", 0),
    ("改善 が 必要 です", 0),
    ("ひどい 対応 でした", 0),
]


def sentence_to_boe(sentence: str, word_to_id: dict[str, int],
                    word_vecs: np.ndarray) -> np.ndarray:
    """Bag-of-Embeddings: mean of word vectors in sentence."""
    H = word_vecs.shape[1]
    vecs = [word_vecs[word_to_id[w]] for w in sentence.split() if w in word_to_id]
    if not vecs:
        return np.zeros(H, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


class LogisticRegression:
    """Simple binary logistic regression with SGD (pure NumPy)."""

    def __init__(self, n_features: int, lr: float = 0.01, n_epochs: int = 200):
        self.W = np.random.randn(n_features).astype(np.float32) * 0.01
        self.b = np.float32(0.0)
        self.lr = lr
        self.n_epochs = n_epochs

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        N = len(X)
        rng = np.random.default_rng(42)
        for _ in range(self.n_epochs):
            idx = rng.permutation(N)
            for i in idx:
                pred = self._sigmoid(float(np.dot(self.W, X[i]) + self.b))
                err = pred - float(y[i])
                self.W = self.W - self.lr * err * X[i]
                self.b = self.b - self.lr * err
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self._sigmoid(X @ self.W + self.b)
        return (probs >= 0.5).astype(np.int32)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


def eval_sentiment(word_vecs: np.ndarray, word_to_id: dict[str, int]) -> dict:
    """Train logistic regression on BoE features, return cross-val accuracy."""
    texts = [d[0] for d in SENTIMENT_DATA]
    labels = np.array([d[1] for d in SENTIMENT_DATA], dtype=np.int32)
    X = np.array([sentence_to_boe(t, word_to_id, word_vecs) for t in texts])

    # 5-fold cross-validation
    N = len(X)
    fold_size = N // 5
    accs = []
    for fold in range(5):
        val_start = fold * fold_size
        val_end   = val_start + fold_size if fold < 4 else N
        val_idx   = list(range(val_start, val_end))
        train_idx = [i for i in range(N) if i not in set(val_idx)]

        X_train, y_train = X[train_idx], labels[train_idx]
        X_val, y_val     = X[val_idx],   labels[val_idx]

        clf = LogisticRegression(n_features=X.shape[1], lr=0.05, n_epochs=100)
        clf.fit(X_train, y_train)
        accs.append(clf.accuracy(X_val, y_val))

    return {
        "cv_accuracy": round(float(np.mean(accs)), 4),
        "cv_std": round(float(np.std(accs)), 4),
        "n_samples": N,
    }


# ===========================================================================
# Main evaluation pipeline
# ===========================================================================

def run_evaluation(n_epochs: int = 150, hidden_size: int = 50,
                   batch_size: int = 64, verbose: bool = True) -> dict:
    corpus, word_to_id, id_to_word = build_vocab_corpus(CORPUS_SENTENCES)
    vocab_size = len(word_to_id)

    if verbose:
        print(f"Corpus: {len(corpus):,} tokens, Vocab: {vocab_size} words")

    contexts, targets = create_contexts_target(corpus, window=2)

    # --- Train models ---
    models_cfg = [
        ("CBOW",     CBOWModel(vocab_size, hidden_size, corpus, 2)),
        ("SkipGram", SkipGramModel(vocab_size, hidden_size, corpus, 2)),
    ]

    all_results: dict[str, dict] = {}

    for model_name, model in models_cfg:
        if verbose:
            print(f"\n=== Training {model_name} ===")
        train_model(model_name, model, contexts, targets,
                    n_epochs=n_epochs, batch_size=batch_size, verbose=verbose)

        if model_name == "Subword":
            word_vecs = model.get_word_vecs()
        else:
            word_vecs = model.word_vecs

        # Normalize vectors
        norms = np.linalg.norm(word_vecs, axis=1, keepdims=True)
        wv_norm = word_vecs / (norms + 1e-8)

        # Evaluate
        analogy_res  = eval_analogy(wv_norm, word_to_id, id_to_word)
        simil_res    = eval_similarity(wv_norm, word_to_id)
        sentiment_res = eval_sentiment(wv_norm, word_to_id)

        all_results[model_name] = {
            "analogy_accuracy": analogy_res["accuracy"],
            "analogy_details": analogy_res,
            "similarity_spearman_rho": simil_res["spearman_rho"],
            "similarity_details": simil_res,
            "sentiment_cv_accuracy": sentiment_res["cv_accuracy"],
            "sentiment_details": sentiment_res,
        }

        if verbose:
            print(f"  Analogy accuracy       : {analogy_res['accuracy']:.4f}"
                  f"  ({analogy_res['correct']}/{analogy_res['valid']})")
            print(f"  Similarity Spearman ρ  : {simil_res['spearman_rho']:.4f}"
                  f"  (n={simil_res['n_pairs']})")
            print(f"  Sentiment accuracy (CV): {sentiment_res['cv_accuracy']:.4f}"
                  f"  ± {sentiment_res['cv_std']:.4f}")

    # Subword model (train for shorter since subword lookup is slow)
    sw_epochs = min(30, n_epochs)
    if verbose:
        print(f"\n=== Training Subword CBOW ({sw_epochs} epochs) ===")
    sw_model = SubwordCBOW(word_to_id, hidden_size, corpus, 2)
    train_model("Subword", sw_model, contexts, targets,
                n_epochs=sw_epochs, batch_size=batch_size, verbose=verbose)
    sw_vecs = sw_model.get_word_vecs()
    norms = np.linalg.norm(sw_vecs, axis=1, keepdims=True)
    sw_vecs_norm = sw_vecs / (norms + 1e-8)

    sw_analogy   = eval_analogy(sw_vecs_norm, word_to_id, id_to_word)
    sw_simil     = eval_similarity(sw_vecs_norm, word_to_id)
    sw_sentiment = eval_sentiment(sw_vecs_norm, word_to_id)

    all_results["Subword"] = {
        "analogy_accuracy": sw_analogy["accuracy"],
        "analogy_details": sw_analogy,
        "similarity_spearman_rho": sw_simil["spearman_rho"],
        "similarity_details": sw_simil,
        "sentiment_cv_accuracy": sw_sentiment["cv_accuracy"],
        "sentiment_details": sw_sentiment,
    }
    if verbose:
        print(f"  Analogy accuracy       : {sw_analogy['accuracy']:.4f}"
              f"  ({sw_analogy['correct']}/{sw_analogy['valid']})")
        print(f"  Similarity Spearman ρ  : {sw_simil['spearman_rho']:.4f}")
        print(f"  Sentiment accuracy (CV): {sw_sentiment['cv_accuracy']:.4f}"
              f"  ± {sw_sentiment['cv_std']:.4f}")

    return all_results


def print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("Word2Vec Model Comparison")
    print("=" * 70)
    header = f"{'Model':<14}  {'Analogy Acc':>12}  {'Simil ρ':>10}  {'Sentiment Acc':>14}"
    print(header)
    print("-" * len(header))
    for model_name, res in results.items():
        print(f"  {model_name:<12}  {res['analogy_accuracy']:>12.4f}"
              f"  {res['similarity_spearman_rho']:>10.4f}"
              f"  {res['sentiment_cv_accuracy']:>14.4f}")
    print("=" * 70)
    print("\nNotes:")
    print("  Analogy Acc   : proportion of a:b=c:? tasks answered correctly")
    print("  Simil ρ       : Spearman rank correlation with human ratings")
    print("  Sentiment Acc : 5-fold CV accuracy on binary sentiment (BoE+LR)")


# ===========================================================================
# __main__
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Word2Vec Evaluation Suite: CBOW vs Skip-gram vs Subword")
    print("=" * 70)

    results = run_evaluation(n_epochs=150, hidden_size=50, verbose=True)
    print_comparison_table(results)

    # Save results (remove detail keys for compactness)
    compact = {
        model: {
            "analogy_accuracy": r["analogy_accuracy"],
            "similarity_spearman_rho": r["similarity_spearman_rho"],
            "sentiment_cv_accuracy": r["sentiment_cv_accuracy"],
        }
        for model, r in results.items()
    }
    out_path = os.path.join(OUT_DIR, "word2vec_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
