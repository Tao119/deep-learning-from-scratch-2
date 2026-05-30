import sys
sys.path.append("..")
import numpy as np
from common.embedding import Embedding, NegativeSamplingLoss
from common.optimizer import Adam
from common.util import preprocess, most_similar
from ch03.cbow import create_contexts_target


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        W_in  = 0.01 * np.random.randn(V, H).astype(np.float32)
        W_out = 0.01 * np.random.randn(V, H).astype(np.float32)

        self.in_layers = [Embedding(W_in) for _ in range(2 * window_size)]
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.ns_loss]
        self.params = []
        self.grads = []
        for l in layers:
            self.params += l.params
            self.grads  += l.grads
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = np.zeros_like(self.in_layers[0].forward(contexts[:, 0]))
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h /= len(self.in_layers)
        return self.ns_loss.forward(h, target)

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout /= len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)


if __name__ == "__main__":
    text = "you say goodbye and i say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    window = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    contexts, target = create_contexts_target(corpus, window=window)
    model = CBOW(vocab_size, hidden_size, window, corpus)
    optimizer = Adam()

    data_size = len(target)
    max_iter = max(1, data_size // batch_size)

    for epoch in range(max_epoch):
        idx = np.random.permutation(data_size)
        ctxs, tgts = contexts[idx], target[idx]
        total = 0
        for i in range(max_iter):
            c = ctxs[i*batch_size:(i+1)*batch_size]
            t = tgts[i*batch_size:(i+1)*batch_size]
            loss = model.forward(c, t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total += loss
        if epoch % 200 == 0:
            print(f"epoch {epoch+1:>5}  loss={total/max_iter:.4f}")

    most_similar("you", word_to_id, id_to_word, model.word_vecs)
    print("\nanalogy: goodbye - you + i =?")
    from common.util import analogy
    analogy("you", "goodbye", "i", word_to_id, id_to_word, model.word_vecs)
