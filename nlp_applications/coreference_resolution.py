"""
nlp_applications/coreference_resolution.py
Simple Neural Coreference Resolution (mention-pair model).

Features:
 - 100 synthetic documents with annotated coreference chains
 - LSTM context encoder + mention feature extractor
 - Binary classifier: coreferent / not coreferent
 - MUC metric evaluation
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field


# ================================================================== #
# Synthetic Data Generation
# ================================================================== #

@dataclass
class Mention:
    start: int          # token index in document
    end: int            # inclusive
    text: str
    gender: str         # M/F/N (neutral)
    number: str         # SG/PL

@dataclass
class Document:
    tokens: list[str]
    mentions: list[Mention]
    clusters: list[list[int]]   # each cluster: list of mention indices


MALE_NAMES   = ["田中", "佐藤", "鈴木", "渡辺", "山田", "中村", "小林", "加藤"]
FEMALE_NAMES = ["花子", "美咲", "由美", "知子", "裕子", "麻衣", "奈緒", "里奈"]
NEUTRAL_NOUNS = ["会社", "学校", "病院", "政府", "委員会", "チーム", "研究所"]


def _gender(name: str) -> str:
    if name in MALE_NAMES:
        return "M"
    if name in FEMALE_NAMES:
        return "F"
    return "N"


def generate_documents(n: int = 100, seed: int = 42) -> list[Document]:
    rng = np.random.default_rng(seed)
    docs: list[Document] = []

    templates = [
        # (token_list, mention_spans, cluster_groups)
        # mention_spans: list of (start, end, text, gender, number)
        # cluster_groups: list of lists of mention indices
    ]

    for doc_id in range(n):
        # Pick entities
        name_m  = rng.choice(MALE_NAMES)
        name_f  = rng.choice(FEMALE_NAMES)
        org     = rng.choice(NEUTRAL_NOUNS)

        # Template 1
        tokens = [
            name_m, "は", org, "に", "勤め", "て", "いる", "。",
            "彼", "は", "毎朝", "そこ", "に", "電車", "で", "通っ", "て", "いる", "。",
            name_f, "も", "同じ", org, "で", "働い", "て", "おり", "、",
            "彼女", "は", name_m, "の", "同僚", "で", "ある", "。",
        ]
        mentions = [
            Mention(0, 0,  name_m,  _gender(name_m), "SG"),  # 0 name_m
            Mention(2, 2,  org,     "N",              "SG"),  # 1 org
            Mention(8, 8,  "彼",   "M",              "SG"),  # 2 彼 → name_m
            Mention(11,11, "そこ", "N",              "SG"),  # 3 そこ → org
            Mention(19,19, name_f,  _gender(name_f), "SG"),  # 4 name_f
            Mention(22,22, org,     "N",              "SG"),  # 5 org (again)
            Mention(28,28, "彼女", "F",              "SG"),  # 6 彼女 → name_f
            Mention(30,30, name_m,  _gender(name_m), "SG"),  # 7 name_m (again)
        ]
        clusters = [[0, 2, 7], [1, 3, 5], [4, 6]]

        docs.append(Document(tokens=tokens, mentions=mentions, clusters=clusters))

    return docs


# ================================================================== #
# Feature Extraction
# ================================================================== #

def extract_mention_pair_features(
    m1: Mention, m2: Mention, m1_idx: int, m2_idx: int,
    vocab: dict[str, int]
) -> np.ndarray:
    """Handcrafted features for a mention pair (m1 precedes m2)."""
    feats = []
    # Distance (bucket: 0,1,2,3-5,6+)
    dist = m2_idx - m1_idx
    dist_bucket = [0.0] * 5
    if   dist == 1:   dist_bucket[0] = 1.0
    elif dist == 2:   dist_bucket[1] = 1.0
    elif dist <= 5:   dist_bucket[2] = 1.0
    elif dist <= 10:  dist_bucket[3] = 1.0
    else:             dist_bucket[4] = 1.0
    feats.extend(dist_bucket)

    # String match
    feats.append(float(m1.text == m2.text))

    # Gender agreement
    if m1.gender == "N" or m2.gender == "N":
        feats.extend([0.0, 1.0])   # unknown
    elif m1.gender == m2.gender:
        feats.extend([1.0, 0.0])
    else:
        feats.extend([0.0, 0.0])

    # Number agreement
    feats.append(float(m1.number == m2.number))

    # Is pronoun
    pronouns = {"彼", "彼女", "彼ら", "それ", "そこ", "その", "この", "ここ"}
    feats.append(float(m1.text in pronouns))
    feats.append(float(m2.text in pronouns))

    # Both proper nouns (not pronoun)
    feats.append(float(m1.text not in pronouns and m2.text not in pronouns))

    return np.array(feats, dtype=np.float64)


FEATURE_DIM = 12  # 5 dist + 1 str + 2 gender + 1 number + 2 pronoun + 1 proper


# ================================================================== #
# LSTM Context Encoder (simplified: encodes mention head word)
# ================================================================== #

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class SimpleLSTM:
    """Minimal LSTM for context encoding (forward pass only, no backprop)."""

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        D, H = input_dim, hidden_dim
        scale = np.sqrt(2.0 / (D + H))
        self.W = rng.standard_normal((4 * H, D + H)) * scale
        self.b = np.zeros(4 * H)
        self.H = H

    def forward(self, sequence: list[np.ndarray]) -> np.ndarray:
        H = self.H
        h = np.zeros(H)
        c = np.zeros(H)
        for x in sequence:
            xh  = np.concatenate([x, h])
            raw = self.W @ xh + self.b
            i = _sigmoid(raw[      :  H])
            f = _sigmoid(raw[  H   :2*H])
            g = np.tanh( raw[2*H   :3*H])
            o = _sigmoid(raw[3*H   :   ])
            c = f * c + i * g
            h = o * np.tanh(c)
        return h


# ================================================================== #
# Coreference Model
# ================================================================== #

class CoreferenceModel:
    """Mention-pair binary classifier."""

    def __init__(self, vocab_size: int, embed_dim: int = 16,
                 lstm_dim: int = 32, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((vocab_size, embed_dim)) * 0.01
        self.lstm = SimpleLSTM(embed_dim, lstm_dim, seed=seed)

        in_dim = 2 * lstm_dim + FEATURE_DIM
        self.W1 = rng.standard_normal((64, in_dim)) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(64)
        self.W2 = rng.standard_normal((1, 64)) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(1)

        # Adam
        for name in ("W1", "b1", "W2", "b2"):
            p = getattr(self, name)
            setattr(self, f"m_{name}", np.zeros_like(p))
            setattr(self, f"v_{name}", np.zeros_like(p))
        self.t = 0

    def _embed_mention(self, mention: Mention, tokens: list[str],
                       vocab: dict[str, int]) -> np.ndarray:
        """Encode the mention context via LSTM on surrounding tokens."""
        start = max(0, mention.start - 3)
        end   = min(len(tokens), mention.end + 4)
        seq   = [self.E[vocab.get(tokens[i], 1)] for i in range(start, end)]
        if not seq:
            seq = [np.zeros(self.E.shape[1])]
        return self.lstm.forward(seq)

    def score(self, m1_enc: np.ndarray, m2_enc: np.ndarray,
              feats: np.ndarray) -> float:
        x  = np.concatenate([m1_enc, m2_enc, feats])
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)
        logit = (self.W2 @ h1 + self.b2)[0]
        return float(_sigmoid(np.array([logit]))[0])

    def _forward_backward(self, m1_enc: np.ndarray, m2_enc: np.ndarray,
                          feats: np.ndarray, label: int) -> tuple[float, np.ndarray]:
        x  = np.concatenate([m1_enc, m2_enc, feats])
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)
        logit = (self.W2 @ h1 + self.b2)[0]
        prob  = float(_sigmoid(np.array([logit]))[0])

        # BCE loss
        eps = 1e-9
        loss = -(label * np.log(prob + eps) + (1 - label) * np.log(1 - prob + eps))

        # Gradients
        d_logit = prob - label
        self.dW2 = d_logit * h1.reshape(1, -1)
        self.db2 = np.array([d_logit])
        dh1 = d_logit * self.W2[0]
        dh1 *= (h1 > 0).astype(float)
        self.dW1 = np.outer(dh1, x)
        self.db1 = dh1

        return loss, None   # encoder not backpropped (simplification)

    def _adam(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self.t += 1
        for name in ("W1", "b1", "W2", "b2"):
            param = getattr(self, name)
            grad  = getattr(self, f"d{name}")
            m     = getattr(self, f"m_{name}")
            v     = getattr(self, f"v_{name}")
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * grad ** 2
            mh = m / (1 - beta1 ** self.t)
            vh = v / (1 - beta2 ** self.t)
            param -= lr * mh / (np.sqrt(vh) + eps)


# ================================================================== #
# MUC Metric
# ================================================================== #

def muc_metric(pred_clusters: list[list[int]], gold_clusters: list[list[int]],
               n_mentions: int) -> tuple[float, float, float]:
    """Simplified MUC recall / precision / F1."""
    def _links(clusters: list[list[int]]) -> set[frozenset]:
        links: set[frozenset] = set()
        for cl in clusters:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    links.add(frozenset({cl[i], cl[j]}))
        return links

    gold_links = _links(gold_clusters)
    pred_links = _links(pred_clusters)

    if not gold_links and not pred_links:
        return 1.0, 1.0, 1.0

    true_pos = len(gold_links & pred_links)
    rec  = true_pos / max(len(gold_links), 1)
    prec = true_pos / max(len(pred_links), 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


# ================================================================== #
# Training & Evaluation
# ================================================================== #

def build_vocab(docs: list[Document]) -> dict[str, int]:
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for doc in docs:
        for tok in doc.tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def generate_pairs(doc: Document) -> list[tuple[int, int, int]]:
    """Return list of (m1_idx, m2_idx, label)."""
    coref_pairs: set[tuple[int, int]] = set()
    for cl in doc.clusters:
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                coref_pairs.add((cl[i], cl[j]))

    pairs = []
    M = len(doc.mentions)
    for i in range(M):
        for j in range(i + 1, M):
            label = 1 if (i, j) in coref_pairs else 0
            pairs.append((i, j, label))
    return pairs


def train(docs: list[Document], epochs: int = 30, lr: float = 0.003,
          seed: int = 0) -> CoreferenceModel:
    vocab = build_vocab(docs)
    model = CoreferenceModel(vocab_size=len(vocab), seed=seed)
    rng   = np.random.default_rng(seed)

    print(f"Vocab size: {len(vocab)} | Documents: {len(docs)}")

    for ep in range(epochs):
        total_loss = 0.0
        n_pairs    = 0
        doc_order  = rng.permutation(len(docs))

        for di in doc_order:
            doc    = docs[di]
            pairs  = generate_pairs(doc)
            if not pairs:
                continue

            # Encode all mentions for this doc
            encs   = [model._embed_mention(m, doc.tokens, vocab) for m in doc.mentions]

            for m1i, m2i, label in pairs:
                feats = extract_mention_pair_features(
                    doc.mentions[m1i], doc.mentions[m2i], m1i, m2i, vocab)
                loss, _ = model._forward_backward(encs[m1i], encs[m2i], feats, label)
                model._adam(lr)
                total_loss += loss
                n_pairs    += 1

        if (ep + 1) % 5 == 0:
            print(f"  Epoch {ep+1:3d}  avg_loss={total_loss / max(n_pairs, 1):.4f}")

    return model


def evaluate(model: CoreferenceModel, docs: list[Document]) -> dict[str, float]:
    vocab = build_vocab(docs)
    all_prec, all_rec, all_f1 = [], [], []

    for doc in docs:
        encs   = [model._embed_mention(m, doc.tokens, vocab) for m in doc.mentions]
        M      = len(doc.mentions)
        THRESHOLD = 0.5

        # Build predicted clusters using greedy antecedent linking
        pred_link: dict[int, int] = {}
        for i in range(M):
            best_score = THRESHOLD
            best_ant   = None
            for j in range(i):
                feats = extract_mention_pair_features(
                    doc.mentions[j], doc.mentions[i], j, i, vocab)
                score = model.score(encs[j], encs[i], feats)
                if score > best_score:
                    best_score = score
                    best_ant   = j
            if best_ant is not None:
                pred_link[i] = best_ant

        # Convert links to clusters (union-find style)
        parent = list(range(M))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for child, ant in pred_link.items():
            rc, ra = find(child), find(ant)
            if rc != ra:
                parent[rc] = ra

        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(M):
            groups[find(i)].append(i)
        pred_clusters = [v for v in groups.values() if len(v) > 1]

        prec, rec, f1 = muc_metric(pred_clusters, doc.clusters, M)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

    return {
        "precision": float(np.mean(all_prec)),
        "recall":    float(np.mean(all_rec)),
        "f1":        float(np.mean(all_f1)),
    }


# ================================================================== #
# Main
# ================================================================== #

def main():
    print("=" * 60)
    print("Coreference Resolution — Mention-pair Neural Model")
    print("=" * 60)

    docs = generate_documents(n=100, seed=42)
    rng  = np.random.default_rng(1)

    idx_all   = rng.permutation(len(docs))
    n_train   = 80
    train_docs = [docs[i] for i in idx_all[:n_train]]
    test_docs  = [docs[i] for i in idx_all[n_train:]]

    print(f"\nTrain: {len(train_docs)} docs | Test: {len(test_docs)} docs")
    print("\nTraining...")
    model = train(train_docs, epochs=30, lr=0.003, seed=42)

    print("\n--- Evaluation on Test Set (MUC Metric) ---")
    metrics = evaluate(model, test_docs)
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  F1        : {metrics['f1']:.3f}")

    print("\n--- Sample Coreference Chains (first 3 test docs) ---")
    vocab = build_vocab(docs)
    for doc in test_docs[:3]:
        encs   = [model._embed_mention(m, doc.tokens, vocab) for m in doc.mentions]
        M      = len(doc.mentions)
        THRESHOLD = 0.5

        pred_link: dict[int, int] = {}
        for i in range(M):
            best_score = THRESHOLD
            best_ant   = None
            for j in range(i):
                feats = extract_mention_pair_features(
                    doc.mentions[j], doc.mentions[i], j, i, vocab)
                score = model.score(encs[j], encs[i], feats)
                if score > best_score:
                    best_score = score
                    best_ant   = j
            if best_ant is not None:
                pred_link[i] = best_ant

        parent = list(range(M))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for child, ant in pred_link.items():
            rc, ra = find(child), find(ant)
            if rc != ra:
                parent[rc] = ra

        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(M):
            groups[find(i)].append(i)
        pred_clusters = [v for v in groups.values() if len(v) > 1]

        print(f"\n  Doc (tokens): {' '.join(doc.tokens[:12])}...")
        print(f"  Gold chains: {[[doc.mentions[i].text for i in cl] for cl in doc.clusters]}")
        print(f"  Pred chains: {[[doc.mentions[i].text for i in cl] for cl in pred_clusters]}")


if __name__ == "__main__":
    main()
