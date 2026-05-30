import sys
sys.path.append("..")
import numpy as np
from common.time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
from common.optimizer import SGD
from common.util import preprocess, clip_grads


class BetterRNNLM:
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        embed_W  = (np.random.randn(V, D) / 100).astype("f")
        lstm_Wx1 = (np.random.randn(D, 4*H) / np.sqrt(D)).astype("f")
        lstm_Wh1 = (np.random.randn(H, 4*H) / np.sqrt(H)).astype("f")
        lstm_b1  = np.zeros(4*H).astype("f")
        lstm_Wx2 = (np.random.randn(H, 4*H) / np.sqrt(H)).astype("f")
        lstm_Wh2 = (np.random.randn(H, 4*H) / np.sqrt(H)).astype("f")
        lstm_b2  = np.zeros(4*H).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.embed  = TimeEmbedding(embed_W)
        self.lstm1  = TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True)
        self.lstm2  = TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True)
        self.affine = TimeAffine(embed_W.T, affine_b)  # weight tying
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params = (self.embed.params + self.lstm1.params
                       + self.lstm2.params + self.affine.params)
        self.grads  = (self.embed.grads  + self.lstm1.grads
                       + self.lstm2.grads  + self.affine.grads)

    def predict(self, xs):
        xs = self.embed.forward(xs)
        xs = self.lstm1.forward(xs)
        xs = self.lstm2.forward(xs)
        return self.affine.forward(xs)

    def forward(self, xs, ts):
        score = self.predict(xs)
        return self.loss_layer.forward(score, ts)

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dout = self.affine.backward(dout)
        dout = self.lstm2.backward(dout)
        dout = self.lstm1.backward(dout)
        self.embed.backward(dout)

    def reset_state(self):
        self.lstm1.reset_state()
        self.lstm2.reset_state()


if __name__ == "__main__":
    text = ("the dog ran . the cat sat . the dog sat . "
            "a cat ran . a dog ran . the cat ran . "
            "the dog ate . a cat ate . the cat ate .")
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size   = len(word_to_id)
    wordvec_size = 64
    hidden_size  = 64
    time_size    = 5
    batch_size   = 4
    max_epoch    = 200
    lr           = 20.0
    max_grad     = 0.25

    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    max_iter  = max(1, data_size // (batch_size * time_size))

    model = BetterRNNLM(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    time_idx = 0

    for epoch in range(max_epoch):
        model.reset_state()
        total_loss = total_count = 0
        for _ in range(max_iter):
            batch_xs = np.zeros((batch_size, time_size), dtype=np.int32)
            batch_ts = np.zeros((batch_size, time_size), dtype=np.int32)
            offsets = [data_size * i // batch_size for i in range(batch_size)]
            for t in range(time_size):
                for b in range(batch_size):
                    batch_xs[b, t] = xs[(offsets[b] + time_idx) % data_size]
                    batch_ts[b, t] = ts[(offsets[b] + time_idx) % data_size]
            time_idx = (time_idx + time_size) % data_size
            loss = model.forward(batch_xs, batch_ts)
            model.backward()
            clip_grads(model.grads, max_grad)
            optimizer.update(model.params, model.grads)
            total_loss += loss
            total_count += 1
        ppl = np.exp(total_loss / total_count)
        if epoch % 50 == 0:
            print(f"epoch {epoch+1:>3}  ppl={ppl:.2f}")

    print(f"\nfinal ppl: {ppl:.2f}")
    print(f"vocab: {word_to_id}")
