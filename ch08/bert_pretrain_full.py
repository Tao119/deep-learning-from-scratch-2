"""
Complete BERT-style Pre-training: MLM + NSP — pure NumPy.

ch08/bert_pretrain_full.py

Architecture
------------
  - Token embedding (vocab_size × d_model)  +  positional encoding
  - [CLS] token prepended to each input
  - 2 × BidirectionalTransformerBlock  (full attention, no causal mask)
  - MLM head:  LayerNorm(d_model) → Linear(d_model → vocab_size)
  - NSP head:  Linear([CLS] d_model → 2)  (binary: isNext / notNext)

Pre-training tasks
------------------
  1. Masked Language Modeling (MLM)
       - Randomly mask 15% of non-special tokens
       - 80% → [MASK], 10% → random token, 10% → unchanged
       - Loss: CrossEntropy on masked positions only

  2. Next Sentence Prediction (NSP)
       - 50% consecutive sentence pairs  (label=1 / isNext)
       - 50% random sentence pairs       (label=0 / notNext)
       - Loss: CrossEntropy on [CLS] position

Training
--------
  100 Japanese sentences corpus (from bert_pretrain.py templates)
  200 epochs, joint loss = MLM_loss + NSP_loss
  Report: MLM accuracy, NSP accuracy per 20 epochs

Fine-tuning demo
----------------
  After pre-training, fine-tune [CLS] representation for sentiment classification
  Compare: pre-trained features vs. random baseline
"""

import sys
import os
import time
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


def _ce_loss(logits, targets):
    """
    logits : (N, C)   N samples, C classes
    targets: (N,)     integer class indices
    Returns scalar loss and softmax probs (N, C)
    """
    probs = _softmax(logits)
    N = len(targets)
    loss = -np.log(probs[np.arange(N), targets] + 1e-7).mean()
    return loss, probs


def clip_grads(grads, max_norm=1.0):
    total = sum(np.sum(g ** 2) for g in grads)
    norm = np.sqrt(total)
    if norm > max_norm:
        rate = max_norm / (norm + 1e-8)
        for g in grads:
            g *= rate


# ---------------------------------------------------------------------------
# Corpus (100 Japanese sentences from template)
# ---------------------------------------------------------------------------

def build_japanese_corpus():
    """
    100 simplified Japanese sentences using medical / everyday templates.
    These are constructed sentences (no real patient data).
    """
    subjects_m = ["患者", "医師", "看護師", "薬剤師", "研修医"]
    subjects_f = ["女性患者", "担当医", "主任看護師", "専門薬剤師", "指導医"]
    actions = [
        "は 病院 に 来院 し た 。",
        "は 診察 を 受け た 。",
        "は 薬 を 服用 し た 。",
        "は 検査 を 受け た 。",
        "は 手術 を 行っ た 。",
        "は 処方 箋 を 書い た 。",
        "は 回診 を 行っ た 。",
        "は 血液 検査 を 依頼 し た 。",
        "は 心電図 を 確認 し た 。",
        "は 入院 手続き を 行っ た 。",
    ]
    diagnoses = [
        "高血圧 は 慢性 疾患 で ある 。",
        "糖尿病 は インスリン 抵抗性 を 引き起こす 。",
        "心筋梗塞 は 緊急 処置 が 必要 だ 。",
        "脳卒中 は 早期 治療 が 重要 だ 。",
        "肺炎 は 抗菌 薬 で 治療 する 。",
        "敗血症 は 生命 の 危険 が ある 。",
        "癌 は 早期 発見 が 大切 だ 。",
        "骨折 は 固定 が 必要 だ 。",
        "喘息 は 気管支 拡張 薬 を 使用 する 。",
        "腎不全 は 透析 が 必要 な 場合 が ある 。",
    ]

    sentences = []
    # Template A: subject + action (50 sentences)
    for subj in subjects_m + subjects_f:
        for act in actions:
            sentences.append(f"{subj} {act}")

    # Template B: diagnosis (10 sentences)
    sentences.extend(diagnoses)

    # Template C: treatment sentences (40 sentences)
    treatments = [
        "アスピリン は 血小板 凝集 を 抑制 する 。",
        "ワルファリン は 抗 凝固 薬 で ある 。",
        "フロセミド は 利尿 作用 が ある 。",
        "アモキシシリン は 抗菌 薬 で ある 。",
        "インスリン は 血糖 を 下げる 働き を する 。",
        "血圧 は 正常 値 以下 に 保つ 必要 が ある 。",
        "体温 は 定期 的 に 測定 する 必要 が ある 。",
        "脈拍 は リズム を 確認 する 必要 が ある 。",
        "酸素 飽和 度 は 九十 五 パーセント 以上 が 正常 で ある 。",
        "呼吸 数 は 安静 時 に 十六 回 程度 が 正常 だ 。",
        "意識 レベル は GCS で 評価 する 。",
        "腎 機能 は クレアチニン で 評価 する 。",
        "肝 機能 は AST と ALT で 評価 する 。",
        "心 機能 は BNP で 評価 する 。",
        "炎症 反応 は CRP で 評価 する 。",
        "感染 症 は 抗菌 薬 で 治療 する 。",
        "脱水 は 輸液 で 治療 する 。",
        "貧血 は 輸血 や 鉄剤 で 治療 する 。",
        "痛み は 鎮痛 薬 で 緩和 する 。",
        "不整脈 は 抗 不整脈 薬 で 治療 する 。",
        "手術 前 に は 絶食 が 必要 で ある 。",
        "術後 は 感染 予防 が 重要 で ある 。",
        "リハビリ は 早期 から 開始 する 。",
        "栄養 管理 は 回復 を 促進 する 。",
        "排泄 管理 は 患者 の 尊厳 を 守る 。",
        "コミュニケーション は 看護 ケア の 基本 で ある 。",
        "インフォームド コンセント は 必須 で ある 。",
        "医療 記録 は 正確 に 記載 する 。",
        "チーム 医療 は 患者 の 安全 を 守る 。",
        "救急 処置 は 迅速 に 行う 必要 が ある 。",
        "予防 接種 は 感染 症 を 防ぐ 。",
        "スクリーニング は 早期 発見 に 役立つ 。",
        "健康 診断 は 定期 的 に 受ける べき だ 。",
        "生活 習慣 病 は 予防 が 大切 で ある 。",
        "禁煙 は 肺 疾患 を 予防 する 。",
        "適度 な 運動 は 健康 を 維持 する 。",
        "バランス の 良い 食事 は 重要 で ある 。",
        "ストレス 管理 は 精神 健康 に 大切 だ 。",
        "睡眠 は 免疫 機能 に 影響 する 。",
        "水分 補給 は 体 の 機能 を 維持 する 。",
    ]
    sentences.extend(treatments)

    return sentences[:100]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]


def build_vocab(sentences):
    PAD, MASK, CLS, SEP, UNK = 0, 1, 2, 3, 4
    all_words = []
    for s in sentences:
        all_words.extend(s.split())
    unique = sorted(set(all_words))
    vocab = SPECIAL_TOKENS + unique
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w, PAD, MASK, CLS, SEP, UNK


def tokenize(sentences, w2i, max_len, PAD, CLS, SEP, UNK):
    """Tokenize with [CLS] at position 0."""
    data = []
    for s in sentences:
        words = s.split()
        ids = [w2i.get(w, UNK) for w in words]
        # Prepend [CLS], append [SEP]
        ids = [CLS] + ids + [SEP]
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [SEP]
        while len(ids) < max_len:
            ids.append(PAD)
        data.append(ids)
    return np.array(data, dtype=np.int32)


# ---------------------------------------------------------------------------
# MLM masking (BERT-style: 80/10/10)
# ---------------------------------------------------------------------------

def apply_bert_mlm_masking(tokens, mask_prob, w2i, vocab_size,
                            PAD, MASK, CLS, SEP):
    """
    BERT-style masking:
      - Skip [PAD], [CLS], [SEP]
      - 15% of remaining tokens are selected
      - 80% → [MASK]
      - 10% → random token
      - 10% → unchanged
    Returns (masked_tokens, labels) where labels[i,j] = original id or -1
    """
    masked = tokens.copy()
    labels = np.full_like(tokens, -1)
    special = {PAD, MASK, CLS, SEP}

    rng = np.random.random(tokens.shape)
    # Select 15% of non-special positions
    is_special = np.isin(tokens, list(special))
    selected = (~is_special) & (rng < mask_prob)

    for r, c in zip(*np.where(selected)):
        original_id = tokens[r, c]
        labels[r, c] = original_id
        p = np.random.random()
        if p < 0.8:
            masked[r, c] = MASK
        elif p < 0.9:
            # Random token (from real vocabulary, not special)
            masked[r, c] = np.random.randint(len(SPECIAL_TOKENS), vocab_size)
        # else: unchanged

    return masked, labels


# ---------------------------------------------------------------------------
# NSP pair construction
# ---------------------------------------------------------------------------

def build_nsp_pairs(sentences, tokens, w2i, max_len, PAD, CLS, SEP, UNK):
    """
    Build sentence pairs for NSP.
    Returns:
        pair_tokens  : (N_pairs, max_len) — [CLS] sentA [SEP] sentB [SEP]
        nsp_labels   : (N_pairs,)  1=isNext, 0=notNext
    """
    N = len(sentences)
    pair_tokens = []
    nsp_labels = []

    for i in range(N):
        # isNext: sentence i and i+1 (wrap around)
        if np.random.random() < 0.5 and i + 1 < N:
            j = i + 1
            label = 1
        else:
            j = np.random.randint(0, N)
            while j == i + 1 or j == i:
                j = np.random.randint(0, N)
            label = 0

        # Tokenize pair: [CLS] A [SEP] B [SEP]
        words_a = sentences[i].split()
        words_b = sentences[j].split()
        ids_a = [w2i.get(w, UNK) for w in words_a]
        ids_b = [w2i.get(w, UNK) for w in words_b]
        ids = [CLS] + ids_a + [SEP] + ids_b + [SEP]

        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [SEP]
        while len(ids) < max_len:
            ids.append(PAD)

        pair_tokens.append(ids)
        nsp_labels.append(label)

    return np.array(pair_tokens, dtype=np.int32), np.array(nsp_labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm:
    def __init__(self, d, eps=1e-6):
        self.gamma = np.ones(d, dtype='f')
        self.beta = np.zeros(d, dtype='f')
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]
        self.eps = eps
        self._cache = None

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        out = self.gamma * xhat + self.beta
        self._cache = (x, xhat, mu, var)
        return out

    def backward(self, dout):
        x, xhat, mu, var = self._cache
        std_inv = 1.0 / np.sqrt(var + self.eps)
        self.grads[0][...] = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
        self.grads[1][...] = dout.sum(axis=tuple(range(dout.ndim - 1)))
        dxhat = dout * self.gamma
        dx = std_inv * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
        return dx


# ---------------------------------------------------------------------------
# Multi-head Attention (full, bidirectional)
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        s = np.sqrt(d_model)
        self.Wq = (np.random.randn(d_model, d_model) / s).astype('f')
        self.Wk = (np.random.randn(d_model, d_model) / s).astype('f')
        self.Wv = (np.random.randn(d_model, d_model) / s).astype('f')
        self.Wo = (np.random.randn(d_model, d_model) / s).astype('f')
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.grads = [np.zeros_like(p) for p in self.params]
        self._cache = None

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
            scores = scores + pad_mask[:, None, None, :] * (-1e9)
        A = _softmax(scores)
        out = self._merge(A @ V) @ self.Wo
        self._cache = (x, Q, K, V, A)
        return out

    def backward(self, dout):
        x, Q, K, V, A = self._cache
        N, T, _ = x.shape
        out_pre = self._merge(A @ V)
        dWo = out_pre.reshape(N * T, self.d_model).T @ dout.reshape(N * T, self.d_model)
        d_merged = dout @ self.Wo.T
        d_AV = self._split(d_merged)
        dA = d_AV @ V.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ d_AV
        dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True)) / np.sqrt(self.d_head)
        dQ = dscores @ K
        dK = dscores.transpose(0, 1, 3, 2) @ Q
        dQ_m, dK_m, dV_m = self._merge(dQ), self._merge(dK), self._merge(dV)
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
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FFN:
    def __init__(self, d_model, d_ff):
        s = np.sqrt(d_model)
        self.W1 = (np.random.randn(d_model, d_ff) / s).astype('f')
        self.b1 = np.zeros(d_ff, dtype='f')
        self.W2 = (np.random.randn(d_ff, d_model) / np.sqrt(d_ff)).astype('f')
        self.b2 = np.zeros(d_model, dtype='f')
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.grads = [np.zeros_like(p) for p in self.params]
        self._cache = None

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        self._cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self._cache
        xr, hr, dr = x.reshape(-1, x.shape[-1]), h.reshape(-1, h.shape[-1]), dout.reshape(-1, dout.shape[-1])
        dW2 = hr.T @ dr
        db2 = dr.sum(0)
        dh = dr @ self.W2.T
        dh[hr == 0] = 0
        dW1 = xr.T @ dh
        db1 = dh.sum(0)
        dx = (dh @ self.W1.T).reshape(x.shape)
        self.grads[0][...] = dW1
        self.grads[1][...] = db1
        self.grads[2][...] = dW2
        self.grads[3][...] = db2
        return dx


# ---------------------------------------------------------------------------
# Encoder Block
# ---------------------------------------------------------------------------

class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.params = (self.norm1.params + self.attn.params
                       + self.norm2.params + self.ffn.params)
        self.grads = (self.norm1.grads + self.attn.grads
                      + self.norm2.grads + self.ffn.grads)
        self._cache = None

    def forward(self, x, pad_mask=None):
        h = x + self.attn.forward(self.norm1.forward(x), pad_mask)
        out = h + self.ffn.forward(self.norm2.forward(h))
        self._cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self._cache
        dh_ffn = self.ffn.backward(self.norm2.backward(dout))
        dh = dout + dh_ffn
        dx_att = self.attn.backward(self.norm1.backward(dh))
        return dh + dx_att


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype='f')
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    if d_model > 1:
        pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe


# ---------------------------------------------------------------------------
# Full BERT Model (MLM + NSP)
# ---------------------------------------------------------------------------

class BertFull:
    """
    Full BERT with MLM + NSP heads.
    """

    def __init__(self, vocab_size, d_model=32, n_heads=2, n_layers=2,
                 d_ff=64, max_len=48):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len

        s = np.sqrt(d_model)
        # Embeddings
        self.embed_W = (np.random.randn(vocab_size, d_model) / s).astype('f')
        self.pe = positional_encoding(max_len, d_model)

        # Transformer blocks
        self.blocks = [EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        # MLM head
        self.mlm_norm = LayerNorm(d_model)
        self.mlm_W = (np.random.randn(d_model, vocab_size) / s).astype('f')
        self.mlm_b = np.zeros(vocab_size, dtype='f')

        # NSP head: simple linear from [CLS] position
        self.nsp_W = (np.random.randn(d_model, 2) / s).astype('f')
        self.nsp_b = np.zeros(2, dtype='f')

        # Collect all params/grads
        self.params = [self.embed_W, self.mlm_W, self.mlm_b, self.nsp_W, self.nsp_b]
        self.grads = [np.zeros_like(p) for p in self.params]
        for blk in self.blocks:
            self.params += list(blk.params)
            self.grads += list(blk.grads)
        self.params += self.mlm_norm.params
        self.grads += self.mlm_norm.grads

        self._cache = None

    def forward(self, xs, mlm_labels, nsp_labels, pad_mask=None):
        """
        xs          : (N, T) masked token ids (with [CLS])
        mlm_labels  : (N, T) original ids at masked positions, -1 elsewhere
        nsp_labels  : (N,)   0/1 NSP labels
        Returns: (total_loss, mlm_loss, nsp_loss)
        """
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]
        self._emb_in = xs

        for blk in self.blocks:
            x = blk.forward(x, pad_mask)

        # MLM head
        h_mlm = self.mlm_norm.forward(x)                     # (N, T, d)
        logits_mlm = h_mlm @ self.mlm_W + self.mlm_b         # (N, T, V)

        mask_pos = mlm_labels != -1
        flat_logits_mlm = logits_mlm[mask_pos]
        flat_labels_mlm = mlm_labels[mask_pos]

        if len(flat_labels_mlm) == 0:
            mlm_loss = 0.0
            mlm_probs = None
        else:
            mlm_loss, mlm_probs = _ce_loss(flat_logits_mlm, flat_labels_mlm)

        # NSP head: use [CLS] token (position 0)
        cls_repr = x[:, 0, :]                                # (N, d)
        logits_nsp = cls_repr @ self.nsp_W + self.nsp_b      # (N, 2)
        nsp_loss, nsp_probs = _ce_loss(logits_nsp, nsp_labels)

        total_loss = mlm_loss + nsp_loss

        self._cache = (x, h_mlm, logits_mlm, mask_pos, flat_labels_mlm, mlm_probs,
                       cls_repr, logits_nsp, nsp_labels, nsp_probs, N, T)
        return total_loss, mlm_loss, nsp_loss

    def backward(self):
        (x, h_mlm, logits_mlm, mask_pos, flat_labels_mlm, mlm_probs,
         cls_repr, logits_nsp, nsp_labels, nsp_probs, N, T) = self._cache

        # Zero accumulate grads for embed
        dW_emb = np.zeros_like(self.embed_W)

        # --- NSP backward ---
        # dL_nsp/d(logits_nsp)
        d_nsp = nsp_probs.copy()
        d_nsp[np.arange(N), nsp_labels] -= 1
        d_nsp /= N

        self.grads[4][...] = d_nsp.sum(axis=0)      # nsp_b grad
        self.grads[3][...] = cls_repr.T @ d_nsp     # nsp_W grad
        d_cls = d_nsp @ self.nsp_W.T                # (N, d)

        # Scatter d_cls back to full sequence gradient (only position 0)
        dx_full = np.zeros((N, T, self.d_model), dtype='f')
        dx_full[:, 0, :] = d_cls

        # --- MLM backward ---
        if mlm_probs is not None and len(flat_labels_mlm) > 0:
            M = len(flat_labels_mlm)
            d_flat = mlm_probs.copy()
            d_flat[np.arange(M), flat_labels_mlm] -= 1
            d_flat /= M

            dlogits_mlm = np.zeros_like(logits_mlm)
            dlogits_mlm[mask_pos] = d_flat

            dh_mlm = dlogits_mlm @ self.mlm_W.T
            self.grads[2][...] = dlogits_mlm.sum(axis=(0, 1))   # mlm_b
            self.grads[1][...] = (
                h_mlm.reshape(N * T, self.d_model).T
                @ dlogits_mlm.reshape(N * T, self.vocab_size)
            )                                                      # mlm_W

            dx_from_mlm = self.mlm_norm.backward(dh_mlm)
            dx_full = dx_full + dx_from_mlm

        # --- Encoder blocks backward ---
        dx = dx_full
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)

        # Embedding gradient
        np.add.at(dW_emb, self._emb_in.reshape(-1), dx.reshape(N * T, self.d_model))
        self.grads[0][...] = dW_emb

    def predict_mlm_accuracy(self, xs, mlm_labels, pad_mask=None):
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]
        for blk in self.blocks:
            x = blk.forward(x, pad_mask)
        h = self.mlm_norm.forward(x)
        logits = h @ self.mlm_W + self.mlm_b
        mask_pos = mlm_labels != -1
        flat_logits = logits[mask_pos]
        flat_labels = mlm_labels[mask_pos]
        if len(flat_labels) == 0:
            return 1.0, 0, 0
        preds = flat_logits.argmax(axis=1)
        correct = int((preds == flat_labels).sum())
        return correct / len(flat_labels), correct, len(flat_labels)

    def predict_nsp_accuracy(self, xs, nsp_labels, pad_mask=None):
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]
        for blk in self.blocks:
            x = blk.forward(x, pad_mask)
        cls_repr = x[:, 0, :]
        logits = cls_repr @ self.nsp_W + self.nsp_b
        preds = logits.argmax(axis=1)
        correct = int((preds == nsp_labels).sum())
        return correct / N, correct, N

    def get_cls_representations(self, xs, pad_mask=None):
        """Forward pass and return [CLS] representations."""
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]
        for blk in self.blocks:
            x = blk.forward(x, pad_mask)
        return x[:, 0, :]  # (N, d_model)


# ---------------------------------------------------------------------------
# Simple Sentiment Classifier (fine-tuning demo)
# ---------------------------------------------------------------------------

class SentimentClassifier:
    """Linear classifier on top of [CLS] features."""

    def __init__(self, d_model, n_classes=2, lr=5e-3):
        s = np.sqrt(d_model)
        self.W = (np.random.randn(d_model, n_classes) / s).astype('f')
        self.b = np.zeros(n_classes, dtype='f')
        self.lr = lr
        self._cache = None

    def forward(self, features, labels):
        logits = features @ self.W + self.b  # (N, 2)
        loss, probs = _ce_loss(logits, labels)
        self._cache = (features, logits, probs, labels)
        return loss, probs

    def backward(self):
        features, logits, probs, labels = self._cache
        N = len(labels)
        d = probs.copy()
        d[np.arange(N), labels] -= 1
        d /= N
        dW = features.T @ d
        db = d.sum(0)
        self.W -= self.lr * dW
        self.b -= self.lr * db

    def accuracy(self, features, labels):
        logits = features @ self.W + self.b
        preds = logits.argmax(axis=1)
        return (preds == labels).mean()


# ---------------------------------------------------------------------------
# Sentiment dataset (synthetic: positive/negative sentences)
# ---------------------------------------------------------------------------

def build_sentiment_data(sentences, w2i, max_len, PAD, CLS, SEP, UNK):
    """
    Assign simple positive/negative labels based on keyword presence.
    Positive: 正常, 良い, 治療, 回復, 正常 → label=1
    Negative: 危険, 障害, 重症, 死, 不全 → label=0
    """
    positive_words = {"正常", "良", "治療", "回復", "予防", "健康", "改善", "成功", "必要", "安全"}
    negative_words = {"危険", "障害", "重症", "死", "不全", "異常", "悪化", "感染", "緊急", "ショック"}

    tokens_list = []
    labels = []

    for s in sentences:
        words = set(s.split())
        pos_score = len(words & positive_words)
        neg_score = len(words & negative_words)
        label = 1 if pos_score >= neg_score else 0

        ids = [CLS] + [w2i.get(w, UNK) for w in s.split()] + [SEP]
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [SEP]
        while len(ids) < max_len:
            ids.append(PAD)
        tokens_list.append(ids)
        labels.append(label)

    return np.array(tokens_list, dtype=np.int32), np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    t0 = time.time()

    print("=" * 65)
    print("Complete BERT-style Pre-training: MLM + NSP — pure NumPy")
    print("=" * 65)

    # Build corpus and vocab
    sentences = build_japanese_corpus()
    print(f"Corpus: {len(sentences)} sentences")
    print(f"Example: '{sentences[0]}'")

    w2i, i2w, PAD, MASK, CLS, SEP, UNK = build_vocab(sentences)
    vocab_size = len(w2i)
    print(f"Vocab size: {vocab_size}  (PAD={PAD}, MASK={MASK}, CLS={CLS})")

    # Determine max length
    max_sentence_len = max(len(s.split()) for s in sentences)
    # Pair: [CLS] A [SEP] B [SEP] — allow up to 2× single sentence + 3 special tokens
    max_len = min(max_sentence_len * 2 + 5, 60)
    max_len_single = max_sentence_len + 3
    print(f"Max token length (pairs): {max_len}, (single): {max_len_single}")

    # Tokenize all sentences (single)
    tokens_single = tokenize(sentences, w2i, max_len_single, PAD, CLS, SEP, UNK)

    # Build NSP pairs
    pair_tokens, nsp_labels_all = build_nsp_pairs(
        sentences, tokens_single, w2i, max_len, PAD, CLS, SEP, UNK
    )
    print(f"NSP pairs: {pair_tokens.shape}  labels: {np.bincount(nsp_labels_all)}")

    # --- Pre-training ---
    model = BertFull(
        vocab_size=vocab_size, d_model=32, n_heads=2,
        n_layers=2, d_ff=64, max_len=max_len
    )
    optimizer = Adam(lr=5e-4)

    EPOCHS = 200
    mask_prob = 0.15
    print(f"\n--- Pre-training: {EPOCHS} epochs, MLM mask_prob={mask_prob} ---")
    print(f"{'Epoch':>6} {'Total':>8} {'MLM':>8} {'NSP':>8} {'MLM_acc':>8} {'NSP_acc':>8}")

    epoch_logs = []

    for epoch in range(1, EPOCHS + 1):
        # Apply MLM masking to pair tokens
        masked_pairs, mlm_labels = apply_bert_mlm_masking(
            pair_tokens, mask_prob, w2i, vocab_size, PAD, MASK, CLS, SEP
        )
        pad_mask = (masked_pairs == PAD).astype('f')

        # Zero all grads
        for g in model.grads:
            g[...] = 0.0

        total_loss, mlm_loss, nsp_loss = model.forward(
            masked_pairs, mlm_labels, nsp_labels_all, pad_mask
        )
        model.backward()
        clip_grads(model.grads, max_norm=1.0)
        optimizer.update(model.params, model.grads)

        if epoch % 20 == 0 or epoch == 1:
            # Evaluate
            mlm_acc, _, _ = model.predict_mlm_accuracy(masked_pairs, mlm_labels, pad_mask)
            nsp_acc, _, _ = model.predict_nsp_accuracy(masked_pairs, nsp_labels_all, pad_mask)
            print(f"{epoch:>6}  {total_loss:>8.4f}  {mlm_loss:>8.4f}  {nsp_loss:>8.4f}"
                  f"  {mlm_acc:>7.1%}  {nsp_acc:>7.1%}")
            epoch_logs.append((epoch, total_loss, mlm_loss, nsp_loss, mlm_acc, nsp_acc))

    elapsed_pretrain = time.time() - t0
    print(f"\nPre-training complete. Runtime: {elapsed_pretrain:.1f}s")

    # Final evaluation
    masked_final, mlm_labels_final = apply_bert_mlm_masking(
        pair_tokens, 0.15, w2i, vocab_size, PAD, MASK, CLS, SEP
    )
    pad_mask_final = (masked_final == PAD).astype('f')
    mlm_acc_final, mlm_correct, mlm_total = model.predict_mlm_accuracy(
        masked_final, mlm_labels_final, pad_mask_final
    )
    nsp_acc_final, nsp_correct, nsp_total = model.predict_nsp_accuracy(
        masked_final, nsp_labels_all, pad_mask_final
    )
    print(f"\nFinal MLM accuracy: {mlm_correct}/{mlm_total} = {mlm_acc_final*100:.1f}%")
    print(f"Final NSP accuracy: {nsp_correct}/{nsp_total} = {nsp_acc_final*100:.1f}%")

    # --- Fine-tuning demo: sentiment classification ---
    print("\n" + "=" * 65)
    print("Fine-tuning Demo: Sentiment Classification")
    print("=" * 65)

    sent_tokens, sent_labels = build_sentiment_data(
        sentences, w2i, max_len_single, PAD, CLS, SEP, UNK
    )
    print(f"Sentiment data: {len(sent_labels)} examples, "
          f"labels: {np.bincount(sent_labels)}")

    # Baseline: random [CLS] features
    print("\n--- Baseline (random features) ---")
    baseline_feat = np.random.randn(len(sent_labels), 32).astype('f')
    clf_base = SentimentClassifier(d_model=32, lr=5e-3)
    for ep in range(200):
        clf_base.forward(baseline_feat, sent_labels)
        clf_base.backward()
    base_acc = clf_base.accuracy(baseline_feat, sent_labels)
    print(f"  Baseline (random features) accuracy: {base_acc*100:.1f}%")

    # Pre-trained features
    print("\n--- Fine-tuning on pre-trained [CLS] features ---")
    pad_mask_sent = (sent_tokens == PAD).astype('f')
    pretrained_feat = model.get_cls_representations(sent_tokens, pad_mask_sent)

    clf_fine = SentimentClassifier(d_model=32, lr=5e-3)
    for ep in range(200):
        clf_fine.forward(pretrained_feat, sent_labels)
        clf_fine.backward()
    fine_acc = clf_fine.accuracy(pretrained_feat, sent_labels)
    print(f"  Pre-trained features accuracy: {fine_acc*100:.1f}%")
    print(f"  Improvement: {(fine_acc - base_acc)*100:+.1f}%")

    # Standard LM comparison
    print("\n--- Comparison: BERT-style vs Standard LM ---")
    print("  BERT-style (MLM+NSP):")
    print(f"    MLM accuracy : {mlm_acc_final*100:.1f}%  (bidirectional context)")
    print(f"    NSP accuracy : {nsp_acc_final*100:.1f}%")
    print(f"    Sentiment    : {fine_acc*100:.1f}% (fine-tuned from [CLS])")
    print("  Standard LM (causal, unidirectional):")
    print("    MLM N/A — uses next-token prediction (see transformer_lm.py)")
    print("    NSP N/A — no sentence-pair task")
    print("  Key advantage of BERT-style: bidirectional attention enables")
    print("  richer contextual representations for classification tasks.")

    total_runtime = time.time() - t0
    print(f"\nTotal runtime: {total_runtime:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
