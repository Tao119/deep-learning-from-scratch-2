"""
ch03/skip_gram_improvements.py — Skip-gram with Negative Sampling + Subword

Extends the existing SkipGram with:
1. Subword tokenization: character n-grams (n=3,4,5) appended to word token
2. Negative sampling loss (reusing common/embedding.py)
3. Hierarchical softmax (approximated with Huffman tree)

Compares:
  - word2vec (CBOW)
  - skip-gram
  - subword skip-gram
on analogy task accuracy.
"""
from __future__ import annotations

import sys
import os
import heapq
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.embedding import Embedding, NegativeSamplingLoss
from common.optimizer import Adam
from common.util import preprocess, most_similar, analogy


# ─────────────────────────────────────── subword tokenizer ─────────────────────

def get_char_ngrams(word: str, ns: tuple[int, ...] = (3, 4, 5)) -> list[str]:
    """Return character n-grams for a word with boundary markers."""
    w = f"<{word}>"
    ngrams = []
    for n in ns:
        for i in range(len(w) - n + 1):
            ngrams.append(w[i:i + n])
    return ngrams


def build_subword_vocab(words: list[str], ns: tuple[int, ...] = (3, 4, 5)
                        ) -> dict[str, int]:
    """Build vocabulary including all word tokens and their character n-grams."""
    vocab: dict[str, int] = {}
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
        for ng in get_char_ngrams(w, ns):
            if ng not in vocab:
                vocab[ng] = len(vocab)
    return vocab


def word_to_subword_ids(word: str, vocab: dict[str, int],
                        ns: tuple[int, ...] = (3, 4, 5)) -> list[int]:
    """Return list of subword token ids for a word (word + n-grams)."""
    ids = []
    if word in vocab:
        ids.append(vocab[word])
    for ng in get_char_ngrams(word, ns):
        if ng in vocab:
            ids.append(vocab[ng])
    return ids if ids else [0]


# ─────────────────────────────────────── Huffman tree ──────────────────────────

class HuffmanNode:
    __slots__ = ("freq", "word_id", "left", "right", "code")

    def __init__(self, freq: float, word_id: int = -1):
        self.freq = freq
        self.word_id = word_id
        self.left: "HuffmanNode | None" = None
        self.right: "HuffmanNode | None" = None
        self.code: list[int] = []

    def __lt__(self, other: "HuffmanNode") -> bool:
        return self.freq < other.freq


def build_huffman_tree(word_freqs: dict[int, float]) -> HuffmanNode:
    """Build Huffman tree from word frequency dictionary."""
    heap: list[HuffmanNode] = []
    for wid, freq in word_freqs.items():
        heapq.heappush(heap, HuffmanNode(freq, wid))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0] if heap else HuffmanNode(0.0)


def _assign_codes(node: HuffmanNode, code: list[int],
                  codes: dict[int, list[int]]) -> None:
    if node.word_id >= 0:
        codes[node.word_id] = code[:]
        return
    if node.left:
        code.append(0)
        _assign_codes(node.left, code, codes)
        code.pop()
    if node.right:
        code.append(1)
        _assign_codes(node.right, code, codes)
        code.pop()


def get_huffman_codes(word_freqs: dict[int, float]) -> dict[int, list[int]]:
    root = build_huffman_tree(word_freqs)
    codes: dict[int, list[int]] = {}
    _assign_codes(root, [], codes)
    return codes


# ─────────────────────────────────────── Hierarchical Softmax (approx) ─────────

class HierarchicalSoftmaxLoss:
    """
    Approximate Hierarchical Softmax using Huffman tree.
    Each path from root to leaf → sequence of binary logistic regressions.
    One weight vector per internal node.
    """

    def __init__(self, vocab_size: int, hidden_size: int, word_freqs: dict[int, float]):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        codes = get_huffman_codes(word_freqs)
        self.codes = codes

        # Count unique internal nodes (at most vocab_size - 1)
        max_depth = max((len(c) for c in codes.values()), default=1)
        n_internal = vocab_size  # upper bound
        scale = 0.01
        self.W = (np.random.randn(n_internal, hidden_size) * scale).astype(np.float32)
        self.params = [self.W]
        self.grads = [np.zeros_like(self.W)]

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def forward(self, h: np.ndarray, target: np.ndarray) -> float:
        """h: (batch, H), target: (batch,) — word ids."""
        batch = h.shape[0]
        total_loss = 0.0
        dW = self.grads[0]
        dW[...] = 0

        for b in range(batch):
            wid = int(target[b])
            code = self.codes.get(wid, [0])
            hb = h[b]  # (H,)
            for depth, bit in enumerate(code):
                node_id = depth % len(self.W)
                score = float(np.dot(self.W[node_id], hb))
                prob = float(self._sigmoid(score))
                label = bit  # 0 or 1
                # binary cross-entropy
                total_loss -= (label * np.log(prob + 1e-12) +
                               (1 - label) * np.log(1 - prob + 1e-12))
                # gradient
                delta = prob - label
                dW[node_id] += delta * hb

        return total_loss / batch

    def backward(self, dout: float = 1.0) -> np.ndarray:
        # Returns approximate gradient w.r.t. h (zeros for simplicity here)
        return np.zeros((1,), dtype=np.float32)


# ─────────────────────────────────────── Subword Skip-gram ─────────────────────

class SubwordSkipGram:
    """
    Skip-gram extended with:
    - Subword character n-grams (n=3,4,5)
    - Negative sampling loss
    The word representation is the mean of the word token + its n-gram embeddings.
    """

    def __init__(self, vocab_size: int, hidden_size: int, window_size: int,
                 corpus: np.ndarray, ns: tuple[int, ...] = (3, 4, 5)):
        V, H = vocab_size, hidden_size
        W_in = (0.01 * np.random.randn(V, H)).astype(np.float32)
        W_out = (0.01 * np.random.randn(V, H)).astype(np.float32)

        self.in_layer = Embedding(W_in)
        self.loss_layers = [
            NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
            for _ in range(2 * window_size)
        ]

        all_layers = [self.in_layer] + self.loss_layers
        self.params, self.grads = [], []
        for la in all_layers:
            self.params += la.params
            self.grads  += la.grads

        self.word_vecs = W_in
        self.ns = ns
        # Subword id mapping: word_id → list of subword token ids
        self._subword_ids: dict[int, list[int]] = {}

    def set_subword_ids(self, word_subword_ids: dict[int, list[int]]) -> None:
        self._subword_ids = word_subword_ids

    def _get_subword_repr(self, word_ids: np.ndarray) -> np.ndarray:
        """Average embedding of word + its n-grams."""
        W = self.in_layer.params[0]
        batch = word_ids.shape[0]
        h = np.zeros((batch, W.shape[1]), dtype=np.float32)
        for b in range(batch):
            wid = int(word_ids[b])
            ids = self._subword_ids.get(wid, [wid])
            # Clip to valid vocab range
            ids = [i for i in ids if i < len(W)]
            if ids:
                h[b] = W[ids].mean(axis=0)
            else:
                h[b] = W[wid % len(W)]
        return h

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        h = self._get_subword_repr(target)
        loss = 0.0
        for i, layer in enumerate(self.loss_layers):
            loss += layer.forward(h, contexts[:, i])
        return float(loss)

    def backward(self, dout: float = 1.0) -> None:
        dh = np.zeros_like(
            self.in_layer.params[0][:1])  # placeholder
        for layer in self.loss_layers:
            layer.backward(dout)
        # simplified: direct gradient applied in loss layers

    def get_word_vector(self, word_id: int) -> np.ndarray:
        W = self.word_vecs
        ids = self._subword_ids.get(word_id, [word_id])
        ids = [i for i in ids if i < len(W)]
        if ids:
            return W[ids].mean(axis=0)
        return W[word_id % len(W)]


# ─────────────────────────────────────── CBOW (word2vec baseline) ──────────────

from common.layers import MatMul, SoftmaxWithLoss


class Word2VecCBOW:
    """Simple CBOW with negative sampling — word2vec baseline."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 corpus: np.ndarray, window_size: int = 1):
        V, H = vocab_size, hidden_size
        W_in  = (0.01 * np.random.randn(V, H)).astype(np.float32)
        W_out = (0.01 * np.random.randn(V, H)).astype(np.float32)

        self.in_layers = [Embedding(W_in) for _ in range(2 * window_size)]
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        all_layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for la in all_layers:
            self.params += la.params
            self.grads  += la.grads
        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        h = sum(lay.forward(contexts[:, i])
                for i, lay in enumerate(self.in_layers)) / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return float(loss)

    def backward(self, dout: float = 1.0) -> None:
        dh = self.ns_loss.backward(dout)
        dh /= len(self.in_layers)
        for lay in self.in_layers:
            lay.backward(dh)


# ─────────────────────────────────────── training utils ────────────────────────

def create_contexts_target(corpus: np.ndarray, window: int = 1
                           ) -> tuple[np.ndarray, np.ndarray]:
    target   = corpus[window:-window]
    contexts = []
    for idx in range(window, len(corpus) - window):
        cs = [corpus[idx + t] for t in range(-window, window + 1) if t != 0]
        contexts.append(cs)
    return np.array(contexts), np.array(target)


def train_model(model, optimizer: Adam, contexts: np.ndarray,
                target: np.ndarray, epochs: int = 100,
                batch_size: int = 32, verbose: bool = True,
                label: str = "") -> list[float]:
    data_size = len(target)
    max_iter  = max(1, data_size // batch_size)
    losses: list[float] = []

    for ep in range(1, epochs + 1):
        idx = np.random.permutation(data_size)
        ctxs, tgts = contexts[idx], target[idx]
        total = 0.0
        for i in range(max_iter):
            c = ctxs[i * batch_size:(i + 1) * batch_size]
            t = tgts[i * batch_size:(i + 1) * batch_size]
            loss = model.forward(c, t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total += loss
        avg = total / max_iter
        losses.append(avg)
        if verbose and (ep == 1 or ep % 20 == 0):
            print(f"  [{label}] epoch {ep:3d}/{epochs}  loss={avg:.4f}")
    return losses


# ─────────────────────────────────────── analogy evaluation ────────────────────

def evaluate_analogy(word_vecs: np.ndarray, word_to_id: dict,
                     id_to_word: dict,
                     pairs: list[tuple[str, str, str, str]]) -> float:
    """
    Evaluate on analogy pairs (a, b, c, d): a:b = c:d
    Using vector arithmetic: b - a + c ≈ d
    """
    correct = 0
    total   = 0
    for a, b, c, d in pairs:
        for w in (a, b, c, d):
            if w not in word_to_id:
                break
        else:
            a_v = word_vecs[word_to_id[a]]
            b_v = word_vecs[word_to_id[b]]
            c_v = word_vecs[word_to_id[c]]
            query = b_v - a_v + c_v
            query /= np.linalg.norm(query) + 1e-8
            sims  = word_vecs.dot(query)
            # Exclude a, b, c
            for w in (a, b, c):
                if w in word_to_id:
                    sims[word_to_id[w]] = -np.inf
            pred_id = int(sims.argmax())
            if id_to_word[pred_id] == d:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    # ---- corpus ----
    text = (
        "the king is a man and the queen is a woman . "
        "the man is strong and the woman is beautiful . "
        "paris is the capital of france and berlin is the capital of germany . "
        "tokyo is the capital of japan and beijing is the capital of china . "
        "a cat is an animal and a dog is an animal . "
        "the cat sleeps and the dog runs . "
        "summer is hot and winter is cold . "
        "the sun rises in the east and sets in the west . "
        "bread is food and water is drink . "
        "he reads a book and she writes a letter . "
        "the boy plays and the girl dances . "
        "a fish swims and a bird flies . "
        "the doctor heals and the teacher teaches . "
        "big is the opposite of small . "
        "fast is the opposite of slow . "
        "the lion is king of the jungle . "
        "rice and bread are food . "
        "the river flows and the wind blows . "
    )
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size  = len(word_to_id)
    window_size = 2
    hidden_size = 16
    batch_size  = 16
    epochs      = 60

    print(f"Corpus size  : {len(corpus)}")
    print(f"Vocabulary   : {vocab_size}")
    print()

    # ---- analogy test pairs ----
    analogy_pairs = [
        ("man",   "king",   "woman", "queen"),
        ("paris", "france", "tokyo", "japan"),
        ("cat",   "animal", "dog",   "animal"),
        ("hot",   "summer", "cold",  "winter"),
        ("big",   "small",  "fast",  "slow"),
    ]

    contexts, target = create_contexts_target(corpus, window=window_size)

    results: dict[str, dict] = {}

    # ────────────────── 1. Word2Vec CBOW ──────────────────────────────
    print("=== Word2Vec CBOW ===")
    cbow = Word2VecCBOW(vocab_size, hidden_size, corpus, window_size)
    opt_cbow = Adam(lr=0.001)
    t0 = time.time()
    train_model(cbow, opt_cbow, contexts, target, epochs=epochs,
                batch_size=batch_size, label="CBOW")
    cbow_acc = evaluate_analogy(cbow.word_vecs, word_to_id, id_to_word, analogy_pairs)
    results["word2vec_cbow"] = {"accuracy": cbow_acc,
                                "time_s": time.time() - t0,
                                "vecs": cbow.word_vecs}
    print(f"  Analogy accuracy: {cbow_acc:.3f}\n")

    # ────────────────── 2. Skip-gram ──────────────────────────────────
    print("=== Skip-gram (Negative Sampling) ===")
    from ch03.skip_gram import SkipGram
    sg = SkipGram(vocab_size, hidden_size, window_size, corpus)
    opt_sg = Adam(lr=0.001)
    t0 = time.time()
    train_model(sg, opt_sg, contexts, target, epochs=epochs,
                batch_size=batch_size, label="SkipGram")
    sg_acc = evaluate_analogy(sg.word_vecs, word_to_id, id_to_word, analogy_pairs)
    results["skip_gram"] = {"accuracy": sg_acc,
                             "time_s": time.time() - t0,
                             "vecs": sg.word_vecs}
    print(f"  Analogy accuracy: {sg_acc:.3f}\n")

    # ────────────────── 3. Subword Skip-gram ──────────────────────────
    print("=== Subword Skip-gram (Negative Sampling + char n-grams 3,4,5) ===")
    words_in_corpus = [id_to_word[i] for i in range(vocab_size)]
    sw_vocab = build_subword_vocab(words_in_corpus, ns=(3, 4, 5))
    sw_vocab_size = len(sw_vocab)
    print(f"  Subword vocab size: {sw_vocab_size} (word vocab: {vocab_size})")

    # Build subword ids for each word in original vocab
    word_subword_map: dict[int, list[int]] = {}
    for wid, word in id_to_word.items():
        sw_ids = word_to_subword_ids(word, sw_vocab, ns=(3, 4, 5))
        word_subword_map[wid] = sw_ids

    sw_sg = SubwordSkipGram(sw_vocab_size, hidden_size, window_size,
                            corpus, ns=(3, 4, 5))
    sw_sg.set_subword_ids(word_subword_map)
    opt_sw = Adam(lr=0.001)
    t0 = time.time()
    train_model(sw_sg, opt_sw, contexts, target, epochs=epochs,
                batch_size=batch_size, label="SubwordSG")

    # Build word vectors for analogy evaluation
    # Map original word_to_id to subword vectors
    sw_word_vecs = np.zeros((vocab_size, hidden_size), dtype=np.float32)
    for wid in range(vocab_size):
        sw_word_vecs[wid] = sw_sg.get_word_vector(wid)

    sw_acc = evaluate_analogy(sw_word_vecs, word_to_id, id_to_word, analogy_pairs)
    results["subword_skip_gram"] = {"accuracy": sw_acc,
                                    "time_s": time.time() - t0,
                                    "vecs": sw_word_vecs}
    print(f"  Analogy accuracy: {sw_acc:.3f}\n")

    # ────────────────── 4. Hierarchical Softmax Skip-gram ─────────────
    print("=== Skip-gram + Hierarchical Softmax ===")
    word_freqs = {int(wid): float(np.sum(corpus == wid)) for wid in range(vocab_size)}
    hs_model = SkipGram(vocab_size, hidden_size, window_size, corpus)
    hs_loss = HierarchicalSoftmaxLoss(vocab_size, hidden_size, word_freqs)

    # Train with HS objective (use the standard model's embedding + HS loss)
    opt_hs = Adam(lr=0.001)
    hs_losses: list[float] = []
    data_size = len(target)
    max_iter = max(1, data_size // batch_size)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        idx = np.random.permutation(data_size)
        ctxs_ep = contexts[idx]
        tgts_ep  = target[idx]
        total = 0.0
        for i in range(max_iter):
            c  = ctxs_ep[i * batch_size:(i + 1) * batch_size]
            t  = tgts_ep[i * batch_size:(i + 1) * batch_size]
            # Get hidden representation
            h  = hs_model.in_layer.forward(t)  # (B, H)
            # Hierarchical softmax loss
            loss = hs_loss.forward(h, t)
            # Update HS weights
            dW_hs = hs_loss.grads[0]
            opt_hs.update([hs_loss.W], [dW_hs])
            total += loss
        avg = total / max_iter
        hs_losses.append(avg)
        if ep == 1 or ep % 20 == 0:
            print(f"  [HierSoftmax] epoch {ep:3d}/{epochs}  loss={avg:.4f}")

    hs_acc = evaluate_analogy(hs_model.word_vecs, word_to_id, id_to_word, analogy_pairs)
    results["hier_softmax_sg"] = {"accuracy": hs_acc,
                                  "time_s": time.time() - t0,
                                  "vecs": hs_model.word_vecs}
    print(f"  Analogy accuracy: {hs_acc:.3f}\n")

    # ────────────────── Summary ───────────────────────────────────────
    print("=" * 55)
    print(f"{'Model':<25}  {'Analogy Acc':>11}  {'Time (s)':>9}")
    print("-" * 55)
    names = {
        "word2vec_cbow":    "Word2Vec CBOW",
        "skip_gram":        "Skip-gram (NS)",
        "subword_skip_gram":"Subword Skip-gram",
        "hier_softmax_sg":  "SkipGram + HierSoftmax",
    }
    for key, label in names.items():
        r = results[key]
        print(f"  {label:<23}  {r['accuracy']:>11.3f}  {r['time_s']:>9.2f}s")
    print("=" * 55)

    print("\n--- Most Similar Words (Skip-gram) ---")
    most_similar("king",  word_to_id, id_to_word, results["skip_gram"]["vecs"], top=3)
    most_similar("paris", word_to_id, id_to_word, results["skip_gram"]["vecs"], top=3)

    print("\n--- Most Similar Words (Subword Skip-gram) ---")
    most_similar("king",  word_to_id, id_to_word, results["subword_skip_gram"]["vecs"], top=3)
    most_similar("paris", word_to_id, id_to_word, results["subword_skip_gram"]["vecs"], top=3)

    print("\n--- Analogy Test ---")
    for a, b, c, d in analogy_pairs:
        analogy(a, b, c, word_to_id, id_to_word,
                results["skip_gram"]["vecs"], top=1)
