import sys
sys.path.append("..")
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar

text = "you say goodbye and i say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size, window=1)

print("corpus:", corpus)
print("id_to_word:", id_to_word)
print("\nco-occurrence matrix (window=1):")
print(C)

most_similar("you", word_to_id, id_to_word, C.astype(np.float32))
