import numpy as np


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = (h * target_W).sum(axis=1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params = self.embed_dot_layers[0].params
        self.grads  = self.embed_dot_layers[0].grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        self.loss = _cross_entropy(np.c_[1 - self.y, self.y], t)
        return self.loss

    def backward(self, dout=1):
        batch = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch
        return dx


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        counts = {}
        for w in corpus:
            counts[w] = counts.get(w, 0) + 1
        vocab_size = len(counts)
        self.vocab_size = vocab_size
        self.word_p = np.zeros(vocab_size)
        for i, c in counts.items():
            self.word_p[i] = c
        self.word_p = (self.word_p ** power) / (self.word_p ** power).sum()

    def get_negative_sample(self, target):
        batch = target.shape[0]
        negative_sample = np.zeros((batch, self.sample_size), dtype=np.int32)
        for i in range(batch):
            p = self.word_p.copy()
            target_idx = int(target[i])
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        return negative_sample


def _cross_entropy(y, t):
    if y.ndim == 1:
        y = y[np.newaxis]
        t = np.array([t])
    if t.ndim == 2:
        t = t.argmax(axis=1)
    batch = y.shape[0]
    return -np.log(y[np.arange(batch), t] + 1e-7).mean()
