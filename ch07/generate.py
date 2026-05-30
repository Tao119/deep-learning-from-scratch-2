import sys
sys.path.append("..")
import numpy as np
from common.time_layers import TimeEmbedding, TimeLSTM, TimeSoftmaxWithLoss
from common.layers import Affine
from common.optimizer import SGD
from common.util import preprocess, clip_grads


class RNNLMGen:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        embed_W = (np.random.randn(V, D) / 100).astype("f")
        lstm_Wx = (np.random.randn(D, 4*H) / np.sqrt(D)).astype("f")
        lstm_Wh = (np.random.randn(H, 4*H) / np.sqrt(H)).astype("f")
        lstm_b  = np.zeros(4*H).astype("f")
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.embed  = TimeEmbedding(embed_W)
        self.lstm   = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = Affine(affine_W, affine_b)
        self.loss_layer = TimeSoftmaxWithLoss()

        from ch06.lstm_lm import BetterRNNLM
        self.params = (self.embed.params + self.lstm.params
                       + [affine_W, affine_b])
        self.grads  = (self.embed.grads  + self.lstm.grads
                       + [np.zeros_like(affine_W), np.zeros_like(affine_b)])

    def predict(self, xs):
        h = self.embed.forward(xs)
        h = self.lstm.forward(h)
        N, T, H = h.shape
        h = h.reshape(N*T, H)
        score = self.affine.forward(h)
        return score.reshape(N, T, -1)

    def forward(self, xs, ts):
        return self.loss_layer.forward(self.predict(xs), ts)

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        N_T, V = dout.shape[0]*dout.shape[1], dout.shape[2]
        dout = dout.reshape(N_T, V)
        dout = self.affine.backward(dout)
        dout = dout.reshape(*self.embed.W.shape[:1:-1], -1).transpose(0, 1, 2)
        self.lstm.backward(dout.reshape(dout.shape[0], -1, dout.shape[-1]))
        self.embed.backward(None)

    def reset_state(self):
        self.lstm.reset_state()

    def generate(self, start_id, skip_ids=None, sample_size=20):
        word_ids = [start_id]
        x = np.array(start_id).reshape(1, 1)
        while len(word_ids) < sample_size:
            h = self.embed.forward(x)
            h = self.lstm.forward(h)
            N, T, H = h.shape
            score = self.affine.forward(h.reshape(N*T, H))
            p = _softmax(score.flatten())
            if skip_ids is not None:
                for sid in skip_ids:
                    p[sid] = 0
                p /= p.sum()
            sampled = np.random.choice(len(p), p=p)
            word_ids.append(sampled)
            x = np.array(sampled).reshape(1, 1)
        return word_ids


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


if __name__ == "__main__":
    text = ("the dog ran . the cat sat . the dog sat . "
            "a cat ran . a dog ran . the cat ran . "
            "the dog ate . a cat ate . the cat ate . " * 5)
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size   = len(word_to_id)
    wordvec_size = 64
    hidden_size  = 128
    time_size    = 5
    batch_size   = 4
    max_epoch    = 400
    lr           = 20.0

    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    max_iter  = max(1, data_size // (batch_size * time_size))

    embed_W = (np.random.randn(vocab_size, wordvec_size) / 100).astype("f")
    lstm_Wx = (np.random.randn(wordvec_size, 4*hidden_size) / np.sqrt(wordvec_size)).astype("f")
    lstm_Wh = (np.random.randn(hidden_size, 4*hidden_size) / np.sqrt(hidden_size)).astype("f")
    lstm_b  = np.zeros(4*hidden_size).astype("f")
    affine_W = (np.random.randn(hidden_size, vocab_size) / np.sqrt(hidden_size)).astype("f")
    affine_b = np.zeros(vocab_size).astype("f")

    embed  = TimeEmbedding(embed_W)
    lstm   = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
    affine_t = __import__("common.time_layers", fromlist=["TimeAffine"]).TimeAffine
    from common.time_layers import TimeAffine, TimeSoftmaxWithLoss

    affine = TimeAffine(affine_W, affine_b)
    loss_layer = TimeSoftmaxWithLoss()
    params = embed.params + lstm.params + affine.params
    grads  = embed.grads  + lstm.grads  + affine.grads
    optimizer = SGD(lr)
    time_idx = 0

    for epoch in range(max_epoch):
        lstm.reset_state()
        total = cnt = 0
        for _ in range(max_iter):
            bxs = np.zeros((batch_size, time_size), dtype=np.int32)
            bts = np.zeros((batch_size, time_size), dtype=np.int32)
            off = [data_size * i // batch_size for i in range(batch_size)]
            for t in range(time_size):
                for b in range(batch_size):
                    bxs[b, t] = xs[(off[b] + time_idx) % data_size]
                    bts[b, t] = ts[(off[b] + time_idx) % data_size]
            time_idx = (time_idx + time_size) % data_size
            h = embed.forward(bxs)
            h = lstm.forward(h)
            score = affine.forward(h)
            loss = loss_layer.forward(score, bts)
            dout = loss_layer.backward()
            dout = affine.backward(dout)
            dout = lstm.backward(dout)
            embed.backward(dout)
            clip_grads(grads, 0.25)
            optimizer.update(params, grads)
            total += loss; cnt += 1

        ppl = np.exp(total / cnt)
        if epoch % 100 == 0:
            print(f"epoch {epoch+1:>4}  ppl={ppl:.2f}")

    print(f"\nfinal ppl: {ppl:.2f}")

    lstm.reset_state()
    start = word_to_id.get("the", 0)
    word_ids = [start]
    x = np.array(start).reshape(1, 1)
    for _ in range(30):
        h = embed.forward(x)
        h = lstm.forward(h)
        score = affine.forward(h)
        p = _softmax(score[0, 0])
        sampled = np.random.choice(len(p), p=p)
        word_ids.append(sampled)
        x = np.array(sampled).reshape(1, 1)
    print("\ngenerated:", " ".join(id_to_word[i] for i in word_ids))
