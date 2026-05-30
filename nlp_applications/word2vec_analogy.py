"""
word2vec_analogy.py

Train CBOW with Negative Sampling on a synthetic Japanese corpus,
then run analogy tasks and visualise word vectors with PCA.

Uses NegativeSamplingLoss from common/embedding.py and Adam from common/optimizer.py.
"""

import sys
import os
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.embedding import NegativeSamplingLoss, EmbeddingDot
from common.layers import MatMul
from common.optimizer import Adam

OUT_DIR = os.path.join(os.path.dirname(__file__), "experiments", "01-sentiment-lstm")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic corpus (500 sentences about Japan)
# ---------------------------------------------------------------------------

RAW_SENTENCES = [
    # Geography / capitals
    "東京 は 日本 の 首都 です",
    "パリ は フランス の 首都 です",
    "ロンドン は イギリス の 首都 です",
    "ベルリン は ドイツ の 首都 です",
    "北京 は 中国 の 首都 です",
    "ソウル は 韓国 の 首都 です",
    "ワシントン は アメリカ の 首都 です",
    "ローマ は イタリア の 首都 です",
    "東京 は 大都市 です",
    "パリ は 大都市 です",
    "日本 の 首都 は 東京 です",
    "フランス の 首都 は パリ です",
    "中国 の 首都 は 北京 です",
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
    "うさぎ は ペット です",
    "ハムスター は ペット です",
    "金魚 は ペット です",
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
    "うさぎ は 草 を 食べ ます",
    "犬 は 忠実 な 動物 です",
    "猫 は 夜行性 の 動物 です",
    "金魚 は 水 の 中 で 泳ぐ",
    "ハムスター は 小さな ペット です",
    # Professions / places of work
    "医者 は 病院 で 働く",
    "先生 は 学校 で 働く",
    "料理人 は レストラン で 働く",
    "警察官 は 交番 で 働く",
    "消防士 は 消防署 で 働く",
    "農家 は 農場 で 働く",
    "弁護士 は 法律事務所 で 働く",
    "エンジニア は 会社 で 働く",
    "研究者 は 大学 で 働く",
    "パイロット は 空港 で 働く",
    "医者 は 患者 を 診る",
    "先生 は 生徒 を 教える",
    "料理人 は 料理 を 作る",
    "警察官 は 街 を 守る",
    "消防士 は 火事 を 消す",
    "農家 は 野菜 を 育てる",
    "弁護士 は 依頼人 を 助ける",
    "エンジニア は プログラム を 書く",
    "研究者 は 研究 を する",
    "パイロット は 飛行機 を 操縦 する",
    # Food
    "寿司 は 日本 の 食べ物 です",
    "ラーメン は 人気 の 食べ物 です",
    "てんぷら は 日本 料理 です",
    "うどん は 麺 料理 です",
    "そば は 日本 の 麺 です",
    "刺身 は 新鮮 な 魚 を 使う",
    "味噌汁 は 毎朝 飲む",
    "寿司 は 魚 と ご飯 で 作る",
    "ラーメン は スープ と 麺 が ある",
    "日本 の 食べ物 は 美味しい",
    # Transport
    "新幹線 は 速い 電車 です",
    "飛行機 は 空 を 飛ぶ",
    "電車 は 駅 に 止まる",
    "バス は 道路 を 走る",
    "自転車 は 人 が こぐ",
    "自動車 は 道路 を 走る",
    "新幹線 は 東京 と 大阪 を 結ぶ",
    "飛行機 は 国際線 に 使われる",
    "電車 は 便利 な 交通手段 です",
    "タクシー は 街 を 走る",
    # Nature / weather
    "春 は 桜 が 咲く 季節 です",
    "夏 は 暑い 季節 です",
    "秋 は 紅葉 が 美しい 季節 です",
    "冬 は 雪 が 降る 季節 です",
    "富士山 は 日本 一 高い 山 です",
    "海 は 広い です",
    "川 は 流れ て いる",
    "森 は 木 が たくさん ある",
    "桜 は 春 に 咲く 花 です",
    "雨 は 水 が 降る",
    # Repeated for more data
    "東京 は 日本 の 大都市 です",
    "犬 は かわいい ペット です",
    "猫 は かわいい ペット です",
    "医者 は 病院 に いる",
    "先生 は 学校 に いる",
    "日本 は 美しい 国 です",
    "フランス は 文化 の 国 です",
    "ライオン は 野生 の 強い 動物 です",
    "電車 は 便利 です",
    "新幹線 は 日本 の 誇り です",
    # Extra filler sentences for density
    "病院 は 患者 が 行く 場所 です",
    "学校 は 生徒 が 行く 場所 です",
    "レストラン は 食事 を する 場所 です",
    "動物 は 自然 に 生き て いる",
    "ペット は 家 で 飼わ れる 動物 です",
    "首都 は 国 の 中心 です",
    "大都市 は 多く の 人 が 住む",
    "国 に は 首都 が ある",
    "日本 の 文化 は 独自 です",
    "日本 の 自然 は 美しい",
]

# Expand corpus to ~500 sentences by repeating with slight variations
_VARIATIONS = [
    "東京 は 大きな 首都 です",
    "パリ は 有名 な 首都 です",
    "犬 は 人間 の 友達 です",
    "猫 は 賢い 動物 です",
    "医者 は 人 の 健康 を 守る",
    "先生 は 子供 に 勉強 を 教える",
    "日本 に は 多く の 動物 が いる",
    "ペット は 家族 の 一員 です",
    "病院 で 医者 が 働く",
    "学校 で 先生 が 教える",
    "東京 と パリ は どちら も 大都市 です",
    "犬 と 猫 は どちら も ペット です",
    "医者 と 先生 は 大切 な 仕事 です",
    "病院 と 学校 は 大切 な 場所 です",
    "日本 は アジア で 有名 な 国 です",
    "首都 は 国 の 政治 の 中心 です",
    "野生 動物 は 自然 の 中 で 暮らす",
    "ペット は 家 の 中 で 暮らす",
    "新幹線 は 速く て 便利 な 電車 です",
    "飛行機 は 速い 乗り物 です",
]

CORPUS_SENTENCES = RAW_SENTENCES + _VARIATIONS * 20
# Trim to 500
CORPUS_SENTENCES = CORPUS_SENTENCES[:500]


# ---------------------------------------------------------------------------
# Vocabulary and corpus
# ---------------------------------------------------------------------------

def build_vocab_and_corpus(sentences):
    word_to_id = {}
    id_to_word = {}
    corpus = []
    for sent in sentences:
        for word in sent.split():
            if word not in word_to_id:
                wid = len(word_to_id)
                word_to_id[word] = wid
                id_to_word[wid] = word
            corpus.append(word_to_id[word])
    return np.array(corpus, dtype=np.int32), word_to_id, id_to_word


# ---------------------------------------------------------------------------
# CBOW model with Negative Sampling
# ---------------------------------------------------------------------------

class CBOW:
    def __init__(self, vocab_size, hidden_size, corpus, window=2):
        V, H = vocab_size, hidden_size
        W_in  = (np.random.randn(V, H) / 100).astype("f")
        W_out = (np.random.randn(V, H) / 100).astype("f")

        self.in_layers = [
            _EmbeddingLayer(W_in)
            for _ in range(2 * window)
        ]
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, sample_size=5)

        self.params = []
        self.grads  = []
        for layer in self.in_layers:
            self.params += layer.params
            self.grads  += layer.grads
        self.params += self.ns_loss.params
        self.grads  += self.ns_loss.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = np.zeros((contexts.shape[0], self.in_layers[0].params[0].shape[1]), dtype="f")
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h /= len(self.in_layers)
        return self.ns_loss.forward(h, target)

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout /= len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)


class _EmbeddingLayer:
    def __init__(self, W):
        self.params = [W]
        self.grads  = [np.zeros_like(W)]
        self._idx = None

    def forward(self, idx):
        W, = self.params
        self._idx = idx
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self._idx, dout)


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def create_contexts_target(corpus, window=2):
    targets  = corpus[window:-window]
    contexts = []
    for i in range(window, len(corpus) - window):
        ctx = [corpus[i - w] for w in range(window, 0, -1)] + \
              [corpus[i + w] for w in range(1, window + 1)]
        contexts.append(ctx)
    return np.array(contexts, dtype=np.int32), np.array(targets, dtype=np.int32)


# ---------------------------------------------------------------------------
# Analogy
# ---------------------------------------------------------------------------

def cosine_similarity(v1, v2, eps=1e-8):
    return np.dot(v1, v2) / (np.linalg.norm(v1) + eps) / (np.linalg.norm(v2) + eps)


def most_similar_words(word, word_to_id, id_to_word, word_vecs, top=5):
    if word not in word_to_id:
        return []
    query = word_vecs[word_to_id[word]]
    sims = [(id_to_word[i], cosine_similarity(query, word_vecs[i]))
            for i in range(len(id_to_word)) if id_to_word[i] != word]
    sims.sort(key=lambda x: -x[1])
    return sims[:top]


def analogy(a, b, c, word_to_id, id_to_word, word_vecs, top=5):
    """a:b = c:?  ->  query = b - a + c"""
    for w in (a, b, c):
        if w not in word_to_id:
            return []
    va = word_vecs[word_to_id[a]]
    vb = word_vecs[word_to_id[b]]
    vc = word_vecs[word_to_id[c]]
    q  = vb - va + vc
    q /= np.linalg.norm(q) + 1e-8
    sims = [(id_to_word[i], float(np.dot(word_vecs[i], q) / (np.linalg.norm(word_vecs[i]) + 1e-8)))
            for i in range(len(id_to_word)) if id_to_word[i] not in (a, b, c)]
    sims.sort(key=lambda x: -x[1])
    return sims[:top]


# ---------------------------------------------------------------------------
# PCA visualisation
# ---------------------------------------------------------------------------

def pca_2d(matrix):
    """Simple 2-component PCA without sklearn."""
    mu = matrix.mean(axis=0)
    X  = matrix - mu
    cov = X.T @ X / len(X)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    return X @ eigvecs[:, order[:2]]


def plot_word_vectors(word_vecs, id_to_word, out_path, n_words=40):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # Try to find a CJK font
        cjk_font = None
        for fp in fm.findSystemFonts():
            name = os.path.basename(fp).lower()
            if any(k in name for k in ("noto", "gothic", "hiragino", "meiryo", "ipag")):
                try:
                    prop = fm.FontProperties(fname=fp)
                    cjk_font = prop
                    break
                except Exception:
                    continue

        vocab_size = len(id_to_word)
        indices = list(range(min(n_words, vocab_size)))
        words   = [id_to_word[i] for i in indices]
        vecs    = word_vecs[indices]
        coords  = pca_2d(vecs.astype(float))

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(coords[:, 0], coords[:, 1], s=20, alpha=0.6)
        for (x, y), w in zip(coords, words):
            fp = cjk_font if cjk_font else None
            ax.annotate(w, (x, y), fontsize=9,
                        fontproperties=fp,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title("Word vectors (PCA 2D)", fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"PCA plot saved: {out_path}")
    except Exception as e:
        print(f"Plot skipped ({e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(0)
    print("=== Word2Vec CBOW with Negative Sampling on Synthetic Japanese Corpus ===\n")

    corpus, word_to_id, id_to_word = build_vocab_and_corpus(CORPUS_SENTENCES)
    vocab_size = len(word_to_id)
    print(f"Corpus tokens: {len(corpus)}  Vocab size: {vocab_size}")

    window     = 2
    hidden_size = 50
    batch_size  = 64
    n_epochs    = 200
    lr          = 1e-3

    contexts, targets = create_contexts_target(corpus, window)
    model     = CBOW(vocab_size, hidden_size, corpus, window)
    optimizer = Adam(lr=lr)

    data_size = len(targets)
    max_iter  = max(1, data_size // batch_size)

    print(f"\nTraining CBOW  epochs={n_epochs}  hidden={hidden_size}  window={window}")
    print(f"{'Epoch':>6}  {'Loss':>8}")
    print("-" * 20)

    for epoch in range(1, n_epochs + 1):
        idx = np.random.permutation(data_size)
        ctxs = contexts[idx]
        tgts = targets[idx]
        total_loss = 0.0
        for it in range(max_iter):
            cb = ctxs[it * batch_size:(it + 1) * batch_size]
            tb = tgts[it * batch_size:(it + 1) * batch_size]
            loss = model.forward(cb, tb)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += float(loss)
        avg_loss = total_loss / max_iter
        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg_loss:>8.4f}")

    word_vecs = model.word_vecs.copy()
    # Normalise
    norms = np.linalg.norm(word_vecs, axis=1, keepdims=True) + 1e-8
    word_vecs_normed = word_vecs / norms

    # ------------------------------------------------------------------
    # Similar words
    # ------------------------------------------------------------------
    print("\n=== Top-5 Similar Words ===")
    similar_results = {}
    for qw in ["東京", "犬", "医者"]:
        top = most_similar_words(qw, word_to_id, id_to_word, word_vecs_normed)
        similar_results[qw] = top
        if not top:
            print(f"  '{qw}' not in vocab")
        else:
            print(f"\n  [{qw}]")
            for w, s in top:
                print(f"    {w}: {s:.4f}")

    # ------------------------------------------------------------------
    # Analogy tasks
    # ------------------------------------------------------------------
    print("\n=== Analogy Tasks ===")
    analogy_tasks = [
        ("東京", "日本", "パリ"),   # 東京:日本 = パリ:?  => フランス
        ("犬", "ペット", "猫"),      # 犬:ペット = 猫:?   => ペット (or similar)
        ("医者", "病院", "先生"),    # 医者:病院 = 先生:? => 学校
    ]
    analogy_results = []
    for a, b, c in analogy_tasks:
        top = analogy(a, b, c, word_to_id, id_to_word, word_vecs_normed)
        print(f"\n  {a}:{b} = {c}:?")
        if not top:
            print("    (some words missing from vocab)")
            analogy_results.append({"query": f"{a}:{b}={c}:?", "answers": []})
        else:
            for w, s in top:
                print(f"    {w}: {s:.4f}")
            analogy_results.append({"query": f"{a}:{b}={c}:?", "answers": [(w, s) for w, s in top]})

    # ------------------------------------------------------------------
    # PCA plot
    # ------------------------------------------------------------------
    pca_path = os.path.join(OUT_DIR, "word_vectors_pca.png")
    plot_word_vectors(word_vecs_normed, id_to_word, pca_path)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "corpus_size": int(len(corpus)),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "n_epochs": n_epochs,
        "similar_words": {k: [(w, float(s)) for w, s in v] for k, v in similar_results.items()},
        "analogy_results": [
            {"query": r["query"], "answers": [(w, float(s)) for w, s in r["answers"]]}
            for r in analogy_results
        ],
    }
    out_path = os.path.join(OUT_DIR, "word2vec_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
