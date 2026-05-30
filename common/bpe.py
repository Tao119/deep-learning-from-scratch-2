"""
Byte-Pair Encoding tokenizer implemented from scratch.

Reference: Sennrich et al. 2016 — "Neural Machine Translation of Rare Words
with Subword Units".
"""

import json
import re
from collections import Counter


def _word_to_chars(word):
    """Split a word into a list of characters with an end-of-word marker."""
    return list(word) + ["</w>"]


def _get_pair_counts(vocab):
    """
    Count adjacent symbol pairs across all words in the vocab.
    vocab: dict  {tuple_of_symbols: frequency}
    """
    counts = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            counts[(symbols[i], symbols[i + 1])] += freq
    return counts


def _merge_pair(pair, vocab):
    """Apply one merge rule to every entry in the vocab."""
    merged = "".join(pair)
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


class BPETokenizer:
    def __init__(self):
        self.merges = []         # list of (str, str) merge rules in order
        self.vocab = {}          # symbol -> int id
        self.inv_vocab = {}      # int id -> symbol

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, texts, vocab_size=500, verbose=False):
        """
        Learn BPE merge rules from a list of strings (or a single string).
        Stops when the vocabulary reaches vocab_size or no pair remains.
        """
        if isinstance(texts, str):
            texts = [texts]

        word_freq: dict[str, int] = Counter()
        for text in texts:
            for word in text.lower().split():
                word_freq[word] += 1

        # Initial vocab: character-level split with </w> marker
        vocab: dict[tuple, int] = {
            tuple(_word_to_chars(word)): freq
            for word, freq in word_freq.items()
        }

        # Seed the symbol set with individual characters
        symbols = set()
        for word_symbols in vocab:
            symbols.update(word_symbols)

        self.merges = []
        step = 0
        while len(symbols) < vocab_size:
            pair_counts = _get_pair_counts(vocab)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.__getitem__)
            vocab = _merge_pair(best_pair, vocab)
            merged = "".join(best_pair)
            self.merges.append(best_pair)
            symbols.add(merged)
            step += 1
            if verbose and step <= 10:
                print(f"merge #{step:>2}: {best_pair[0]!r} + {best_pair[1]!r} -> {merged!r}  "
                      f"(count={pair_counts[best_pair]})")

        # Build final vocabulary with integer ids
        all_symbols = sorted(symbols)
        self.vocab = {s: i for i, s in enumerate(all_symbols)}
        self.inv_vocab = {i: s for s, i in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _apply_merges(self, word):
        """Tokenise one word (already split into chars + </w>) using merge rules."""
        symbols = list(_word_to_chars(word))
        for pair in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    symbols = symbols[:i] + ["".join(pair)] + symbols[i + 2:]
                else:
                    i += 1
        return symbols

    def encode(self, text):
        """Return a list of integer token ids for the input text."""
        ids = []
        for word in text.lower().split():
            for sym in self._apply_merges(word):
                ids.append(self.vocab.get(sym, self.vocab.get("<unk>", 0)))
        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids):
        """Reconstruct text from a list of token ids."""
        tokens = [self.inv_vocab.get(i, "") for i in ids]
        # Join tokens; </w> marks a word boundary
        text = "".join(tokens)
        return text.replace("</w>", " ").strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        data = {
            "merges": [[a, b] for a, b in self.merges],
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.vocab = {k: int(v) for k, v in data["vocab"].items()}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    corpus = "you say goodbye and i say hello . " * 100

    tokenizer = BPETokenizer()

    print("=== BPE Training (first 10 merge steps) ===")
    tokenizer.train(corpus, vocab_size=50, verbose=True)

    print(f"\nFinal vocab size: {len(tokenizer.vocab)}")
    print(f"Vocabulary symbols: {sorted(tokenizer.vocab.keys())}")

    print("\n=== Encode / Decode round-trip ===")
    test_sentence = "you say goodbye and i say hello"
    ids = tokenizer.encode(test_sentence)
    reconstructed = tokenizer.decode(ids)
    print(f"  input     : {test_sentence!r}")
    print(f"  token ids : {ids}")
    print(f"  tokens    : {[tokenizer.inv_vocab[i] for i in ids]}")
    print(f"  decoded   : {reconstructed!r}")
    print(f"  round-trip OK: {reconstructed == test_sentence}")

    print("\n=== Save / Load round-trip ===")
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    tokenizer.save(path)
    tok2 = BPETokenizer()
    tok2.load(path)
    ids2 = tok2.encode(test_sentence)
    print(f"  ids match after reload: {ids == ids2}")
    os.unlink(path)
