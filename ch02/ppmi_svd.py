import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi, most_similar

text = "you say goodbye and i say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

print("PPMI matrix:")
np.set_printoptions(precision=3)
print(W)

U, S, V = np.linalg.svd(W)
print("\nword vectors (top 2 dims after SVD):")
for word, wid in word_to_id.items():
    print(f"  {word:10s}: {U[wid, :2]}")

plt.figure(figsize=(6, 6))
for word, wid in word_to_id.items():
    plt.scatter(U[wid, 0], U[wid, 1])
    plt.annotate(word, (U[wid, 0], U[wid, 1]),
                 textcoords="offset points", xytext=(5, 2))
plt.title("Word Vectors (SVD, 2D)")
plt.tight_layout()
plt.savefig("word_vectors_2d.png", dpi=120)
print("\nsaved: word_vectors_2d.png")
most_similar("you", word_to_id, id_to_word, U)
