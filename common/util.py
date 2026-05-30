import sys
import numpy as np


def preprocess(text):
    text = text.lower().replace(".", " .").replace(",", " ,")
    words = text.split()
    word_to_id = {}
    id_to_word = {}
    for w in words:
        if w not in word_to_id:
            nid = len(word_to_id)
            word_to_id[w] = nid
            id_to_word[nid] = w
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window=1):
    T = len(corpus)
    C = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window + 1):
            if idx - i >= 0:
                C[word_id, corpus[idx - i]] += 1
            if idx + i < T:
                C[word_id, corpus[idx + i]] += 1
    return C


def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = C.sum()
    S = C.sum(axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)
            cnt += 1
            if verbose and cnt % (total // 100 + 1) == 0:
                print(f"\r{100*cnt/total:.1f}% done", end="")
    if verbose:
        print()
    return M


def cos_similarity(x, y, eps=1e-8):
    return np.dot(x, y) / (np.linalg.norm(x) + eps) / (np.linalg.norm(y) + eps)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f"'{query}' not found")
        return
    print(f"\n[query] {query}")
    query_vec = word_matrix[word_to_id[query]]
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f" {id_to_word[i]}: {similarity[i]:.4f}")
        count += 1
        if count >= top:
            break


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5):
    for word in (a, b, c):
        if word not in word_to_id:
            print(f"'{word}' not found")
            return
    print(f"\n[analogy] {a}:{b} = {c}:?")
    a_vec = word_matrix[word_to_id[a]]
    b_vec = word_matrix[word_to_id[b]]
    c_vec = word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    similarity = word_matrix.dot(query_vec)
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] in (a, b, c):
            continue
        print(f" {id_to_word[i]}: {similarity[i]:.4f}")
        count += 1
        if count >= top:
            break


def convert_one_hot(corpus, vocab_size):
    N = len(corpus)
    one_hot = np.zeros((N, vocab_size), dtype=np.int32)
    for i, word_id in enumerate(corpus):
        one_hot[i, word_id] = 1
    return one_hot


def clip_grads(grads, max_norm):
    total_norm = 0
    for g in grads:
        total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for g in grads:
            g *= rate
