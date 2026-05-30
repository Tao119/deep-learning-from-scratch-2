"""
sentiment_lstm.py

Character-level LSTM for Japanese movie review sentiment classification.
Uses the TimeLSTM layer from common/time_layers.py directly.

Architecture: char Embedding -> TimeLSTM -> mean-pool -> Affine(2) -> Softmax+Loss
"""

import sys
import os
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.time_layers import TimeLSTM, TimeEmbedding
from common.layers import Affine, SoftmaxWithLoss
from common.optimizer import Adam

OUT_DIR = os.path.join(os.path.dirname(__file__), "experiments", "01-sentiment-lstm")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic Japanese movie review dataset (200 examples, 50/50 pos/neg)
# ---------------------------------------------------------------------------

POSITIVE_REVIEWS = [
    "この映画は感動的で素晴らしかった",
    "演技が上手く涙が出た",
    "ストーリーが深く心に刺さった",
    "映像美が圧倒的で見惚れた",
    "音楽が素晴らしく気持ちが高まった",
    "キャストの演技力が見事だった",
    "脚本が巧みで伏線の回収が見事",
    "感情移入できて最後まで楽しめた",
    "監督のセンスが光る傑作",
    "久しぶりに映画館で泣いた",
    "主人公の成長が感動的だった",
    "テンポが良く飽きずに見られた",
    "予想外の展開に驚かされた",
    "エンディングが美しく余韻が残った",
    "全キャラクターに愛着が湧いた",
    "何度でも見返したい名作",
    "細部まで丁寧に作られた作品",
    "友達にも強くお薦めしたい",
    "公開初日に見て良かった",
    "人生観が変わるような映画",
    "笑いあり涙ありで充実した二時間",
    "これほど完成度が高い映画は久しぶり",
    "続編が待ち遠しい",
    "役者全員が輝いていた",
    "シナリオのクオリティが突出していた",
    "映像と音楽の融合が見事だった",
    "心が温まるヒューマンドラマ",
    "社会的メッセージが重く、考えさせられた",
    "子供と一緒に楽しめる良作",
    "アクションシーンが迫力満点",
    "コメディとシリアスのバランスが絶妙",
    "主題歌が映画にピッタリ合っていた",
    "期待以上の出来で大満足",
    "見終わった後に清々しい気持ちになった",
    "登場人物の感情が丁寧に描かれていた",
    "映画の世界観に引き込まれた",
    "クライマックスが鳥肌ものだった",
    "家族愛が胸に響いた",
    "友情の描写がリアルで共感できた",
    "監督の演出が巧みで飽きが来ない",
    "特殊効果が現実感あふれていた",
    "終盤の展開が秀逸だった",
    "伏線の張り方と回収が完璧",
    "見る価値のある一本",
    "思わず拍手したくなる結末",
    "感情を揺さぶる演技の数々",
    "この映画があって良かった",
    "記憶に残る映像表現",
    "脇役も存在感があって良かった",
    "総合的に言って傑作だと思う",
]

NEGATIVE_REVIEWS = [
    "退屈でつまらない映画",
    "ストーリーが弱く失望した",
    "期待外れで時間の無駄",
    "主人公の行動が理解できない",
    "脚本に矛盾が多すぎる",
    "演技が棒読みで感情移入できない",
    "終わり方が意味不明だった",
    "テンポが遅く眠くなった",
    "キャラクターに魅力がなかった",
    "映像が安っぽく見るに堪えない",
    "伏線が回収されないまま終わった",
    "お金を払う価値がなかった",
    "設定が非現実的すぎる",
    "あんなにひどい映画は初めて",
    "途中で帰りたくなった",
    "音楽がシーンに合っていなかった",
    "説明が少なすぎて話についていけない",
    "陳腐なクリシェばかりで新鮮味がない",
    "二時間があっという間に溝に消えた",
    "主人公に共感できず最後まで苦痛だった",
    "監督のセンスのなさが際立つ",
    "CGが安っぽくて興ざめした",
    "台詞が大げさで不自然",
    "物語の焦点が定まらない散漫な作品",
    "登場人物が誰も好きになれなかった",
    "見終わった後に後悔した",
    "展開が予想通りすぎてつまらない",
    "感動を狙いすぎて逆効果だった",
    "原作ファンとして許せない改変",
    "音響が悪くセリフが聞き取れなかった",
    "アクションシーンが嘘くさい",
    "全体的に完成度が低い",
    "前作と比べて明らかに質が落ちた",
    "キャスティングが間違っている",
    "長すぎて後半は集中できなかった",
    "笑いのセンスが古くスベりまくり",
    "感情の起伏がなく平坦な映画",
    "悪役の動機が薄すぎる",
    "脇役の扱いがぞんざいすぎた",
    "クライマックスが拍子抜けだった",
    "同じような場面の繰り返しで飽き飽きした",
    "せっかくの題材を台無しにした",
    "粗雑な編集が目立ちリズムが悪い",
    "最後まで見て後悔した",
    "登場人物の言動が一貫していない",
    "こんな作品に出資した人に同情する",
    "どこかで見たような話の使い回し",
    "映画と呼ぶのも憚られる出来",
    "作り手の熱意がまったく感じられない",
    "二度と見たくない",
]

DEMO_REVIEWS = [
    "映画が素晴らしく感動した",
    "演技が下手でつまらなかった",
    "ストーリーが面白く楽しめた",
    "退屈で眠くなってしまった",
    "心に残る素晴らしい作品だった",
]


# ---------------------------------------------------------------------------
# Tokenization and vocabulary
# ---------------------------------------------------------------------------

def build_char_vocab(texts):
    chars = set()
    for t in texts:
        chars.update(t)
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 = padding
    vocab["<PAD>"] = 0
    return vocab


def encode(text, vocab, max_len):
    ids = [vocab.get(c, 0) for c in text]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return np.array(ids, dtype=np.int32)


def encode_batch(texts, vocab, max_len):
    return np.stack([encode(t, vocab, max_len) for t in texts])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SentimentLSTM:
    """Embedding -> TimeLSTM -> mean-pool -> Affine(2) -> SoftmaxWithLoss"""

    def __init__(self, vocab_size, embed_dim, hidden_size):
        V, D, H = vocab_size, embed_dim, hidden_size
        embed_W = (np.random.randn(V, D) / 100).astype("f")
        lstm_Wx = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b  = np.zeros(4 * H, dtype="f")
        affine_W = (np.random.randn(H, 2) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(2, dtype="f")

        self.embed  = _EmbeddingLayer(embed_W)
        self.lstm   = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        self.affine = Affine(affine_W, affine_b)
        self.loss_layer = SoftmaxWithLoss()

        self.params = (self.embed.params + self.lstm.params + self.affine.params)
        self.grads  = (self.embed.grads  + self.lstm.grads  + self.affine.grads)

        self._cache = None

    def predict(self, xs):
        emb = self.embed.forward(xs)         # (N, T, D)
        hs  = self.lstm.forward(emb)         # (N, T, H)
        mask = (xs != 0).astype("f")        # (N, T)  1=valid, 0=pad
        mask_exp = mask[:, :, np.newaxis]    # (N, T, 1)
        pooled = (hs * mask_exp).sum(axis=1) / (mask_exp.sum(axis=1) + 1e-8)  # (N, H)
        self._cache = (hs, mask_exp, mask_exp.sum(axis=1))
        logits = self.affine.forward(pooled) # (N, 2)
        return logits

    def forward(self, xs, ts):
        logits = self.predict(xs)
        return self.loss_layer.forward(logits, ts)

    def backward(self):
        dlogits = self.loss_layer.backward()   # (N, 2)
        dpooled = self.affine.backward(dlogits) # (N, H)
        hs, mask_exp, denom = self._cache
        # backprop through mean-pool: dhs = dpooled / denom * mask_exp
        dhs = (dpooled[:, np.newaxis, :] / denom[:, np.newaxis, :]) * mask_exp  # (N, T, H)
        demb = self.lstm.backward(dhs)
        self.embed.backward(demb)


class _EmbeddingLayer:
    """Simple (non-time) embedding layer that returns (N, T, D)."""

    def __init__(self, W):
        self.params = [W]
        self.grads  = [np.zeros_like(W)]
        self._xs = None

    def forward(self, xs):
        W, = self.params
        self._xs = xs
        return W[xs]   # (N, T, D)

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self._xs, dout)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def accuracy(logits, labels):
    preds = logits.argmax(axis=1)
    return (preds == labels).mean()


def train(model, optimizer, xs, ts, batch_size):
    n = len(xs)
    idx = np.random.permutation(n)
    xs, ts = xs[idx], ts[idx]
    total_loss = 0.0
    n_batches = 0
    for start in range(0, n, batch_size):
        xb = xs[start:start + batch_size]
        tb = ts[start:start + batch_size]
        model.lstm.reset_state()
        loss = model.forward(xb, tb)
        model.backward()
        _clip_grads(model.grads, 5.0)
        optimizer.update(model.params, model.grads)
        total_loss += float(loss)
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, xs, ts, batch_size=32):
    all_logits = []
    for start in range(0, len(xs), batch_size):
        xb = xs[start:start + batch_size]
        model.lstm.reset_state()
        logits = model.predict(xb)
        all_logits.append(logits)
    logits = np.vstack(all_logits)
    return accuracy(logits, ts)


def _clip_grads(grads, max_norm):
    total = sum(np.sum(g ** 2) for g in grads) ** 0.5
    rate = max_norm / (total + 1e-6)
    if rate < 1:
        for g in grads:
            g *= rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    print("=== Sentiment LSTM: Japanese Movie Review Classifier ===\n")

    texts  = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
    labels = np.array([1] * len(POSITIVE_REVIEWS) + [0] * len(NEGATIVE_REVIEWS), dtype=np.int32)

    idx = np.random.permutation(len(texts))
    texts  = [texts[i] for i in idx]
    labels = labels[idx]

    split = int(0.8 * len(texts))
    train_texts, val_texts   = texts[:split], texts[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    vocab = build_char_vocab(texts)
    vocab_size = len(vocab)
    max_len = max(len(t) for t in texts)

    print(f"Vocab size (chars): {vocab_size}")
    print(f"Max sequence length: {max_len}")
    print(f"Train: {len(train_texts)}  Val: {len(val_texts)}\n")

    X_train = encode_batch(train_texts, vocab, max_len)
    X_val   = encode_batch(val_texts, vocab, max_len)

    embed_dim   = 32
    hidden_size = 64
    batch_size  = 16
    n_epochs    = 50
    lr          = 1e-3

    model     = SentimentLSTM(vocab_size, embed_dim, hidden_size)
    optimizer = Adam(lr=lr)

    history = []
    print(f"{'Epoch':>5}  {'Loss':>8}  {'Train Acc':>10}  {'Val Acc':>8}")
    print("-" * 40)

    for epoch in range(1, n_epochs + 1):
        loss = train(model, optimizer, X_train, train_labels, batch_size)
        train_acc = evaluate(model, X_train, train_labels)
        val_acc   = evaluate(model, X_val, val_labels)
        history.append({"epoch": epoch, "loss": loss, "train_acc": float(train_acc), "val_acc": float(val_acc)})

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>5}  {loss:>8.4f}  {train_acc:>10.4f}  {val_acc:>8.4f}")

    print()

    # ------------------------------------------------------------------
    # Demo: classify new sentences
    # ------------------------------------------------------------------
    print("=== Demo: Classify new reviews ===")
    label_names = {0: "negative", 1: "positive"}

    model.lstm.reset_state()
    X_demo = encode_batch(DEMO_REVIEWS, vocab, max_len)
    logits = model.predict(X_demo)
    probs  = _softmax(logits)

    demo_results = []
    for review, p in zip(DEMO_REVIEWS, probs):
        pred = int(p.argmax())
        conf = float(p[pred])
        print(f"  [{label_names[pred]} ({conf:.2%})] {review}")
        demo_results.append({"review": review, "prediction": label_names[pred], "confidence": conf})

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "vocab_size": vocab_size,
        "max_seq_len": int(max_len),
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "n_epochs": n_epochs,
        "final_train_acc": history[-1]["train_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "history": history,
        "demo_results": demo_results,
    }
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_path}")
    print(f"Final train acc: {history[-1]['train_acc']:.4f}")
    print(f"Final val acc:   {history[-1]['val_acc']:.4f}")


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()
