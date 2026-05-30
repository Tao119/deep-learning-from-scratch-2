import sys
sys.path.append("..")
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss
from common.util import preprocess, create_co_matrix, most_similar, analogy


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in  = 0.01 * np.random.randn(V, H).astype(np.float32)
        W_out = 0.01 * np.random.randn(H, V).astype(np.float32)
        self.in_layer0  = MatMul(W_in)
        self.in_layer1  = MatMul(W_in)
        self.out_layer  = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params = []
        self.grads = []
        for l in layers:
            self.params += l.params
            self.grads  += l.grads
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) / 2
        score = self.out_layer.forward(h)
        return self.loss_layer.forward(score, target)

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da /= 2
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)


def create_contexts_target(corpus, window=1):
    target = corpus[window:-window]
    contexts = []
    for idx in range(window, len(corpus) - window):
        cs = []
        for t in range(-window, window + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    N = len(corpus)
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for i, w in enumerate(corpus):
            one_hot[i, w] = 1
        return one_hot
    C = corpus.shape[1]
    one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
    for i, cs in enumerate(corpus):
        for j, w in enumerate(cs):
            one_hot[i, j, w] = 1
    return one_hot


if __name__ == "__main__":
    from common.optimizer import Adam

    text = "you say goodbye and i say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    contexts, target = create_contexts_target(corpus, window=1)
    contexts_oh = convert_one_hot(contexts, vocab_size)
    target_oh   = convert_one_hot(target, vocab_size)

    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()
    data_size = len(contexts)
    max_iter = max(1, data_size // batch_size)
    loss_list = []

    for epoch in range(max_epoch):
        idx = np.random.permutation(data_size)
        ctxs = contexts_oh[idx]
        tgts = target[idx]
        total_loss = 0
        for i in range(max_iter):
            c = ctxs[i*batch_size:(i+1)*batch_size]
            t = tgts[i*batch_size:(i+1)*batch_size]
            loss = model.forward(c, t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
        avg = total_loss / max_iter
        loss_list.append(avg)
        if epoch % 200 == 0:
            print(f"epoch {epoch+1:>5}  loss={avg:.4f}")

    word_vecs = model.word_vecs
    print("\nword vectors:")
    for word, wid in word_to_id.items():
        print(f"  {word:10s}: {word_vecs[wid]}")
    most_similar("you", word_to_id, id_to_word, word_vecs)
