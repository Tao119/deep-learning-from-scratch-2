import sys
sys.path.append("..")
import numpy as np
from common.time_layers import TimeEmbedding, TimeGRU, TimeAffine, TimeSoftmaxWithLoss
from common.optimizer import SGD
from common.util import preprocess, clip_grads


class GRULM:
    """
    GRU-based language model with two stacked GRU layers and weight tying.
    Architecture mirrors lstm_lm.py but replaces TimeLSTM with TimeGRU.
    """

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        def gru_params(in_dim, out_dim):
            Wx_zr = (np.random.randn(in_dim, 2*out_dim) / np.sqrt(in_dim)).astype("f")
            Wx_h  = (np.random.randn(in_dim,   out_dim) / np.sqrt(in_dim)).astype("f")
            Wh_zr = (np.random.randn(out_dim, 2*out_dim) / np.sqrt(out_dim)).astype("f")
            Wh_h  = (np.random.randn(out_dim,   out_dim) / np.sqrt(out_dim)).astype("f")
            b_zr  = np.zeros(2*out_dim).astype("f")
            b_h   = np.zeros(out_dim).astype("f")
            return Wx_zr, Wx_h, Wh_zr, Wh_h, b_zr, b_h

        embed_W  = (np.random.randn(V, D) / 100).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.embed  = TimeEmbedding(embed_W)
        self.gru1   = TimeGRU(*gru_params(D, H), stateful=True)
        self.gru2   = TimeGRU(*gru_params(H, H), stateful=True)
        self.affine = TimeAffine(embed_W.T, affine_b)   # weight tying
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params = (self.embed.params + self.gru1.params
                       + self.gru2.params + self.affine.params)
        self.grads  = (self.embed.grads  + self.gru1.grads
                       + self.gru2.grads  + self.affine.grads)

    def predict(self, xs):
        xs = self.embed.forward(xs)
        xs = self.gru1.forward(xs)
        xs = self.gru2.forward(xs)
        return self.affine.forward(xs)

    def forward(self, xs, ts):
        return self.loss_layer.forward(self.predict(xs), ts)

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dout = self.affine.backward(dout)
        dout = self.gru2.backward(dout)
        dout = self.gru1.backward(dout)
        self.embed.backward(dout)

    def reset_state(self):
        self.gru1.reset_state()
        self.gru2.reset_state()


def train(model, corpus, batch_size, time_size, max_epoch, lr, max_grad, label):
    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    max_iter  = max(1, data_size // (batch_size * time_size))
    optimizer = SGD(lr)
    time_idx  = 0
    ppl_list  = []

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
        ppl_list.append(float(ppl))
        if epoch % 50 == 0:
            print(f"[{label}] epoch {epoch+1:>3}  ppl={ppl:.2f}")
    return ppl_list


if __name__ == "__main__":
    from common.time_layers import TimeLSTM

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

    # ---- GRU language model ----
    gru_model = GRULM(vocab_size, wordvec_size, hidden_size)
    gru_ppls  = train(gru_model, corpus, batch_size, time_size,
                      max_epoch, lr, max_grad, "GRU")

    # ---- LSTM language model for comparison ----
    from ch06.lstm_lm import BetterRNNLM
    lstm_model = BetterRNNLM(vocab_size, wordvec_size, hidden_size)
    lstm_ppls  = train(lstm_model, corpus, batch_size, time_size,
                       max_epoch, lr, max_grad, "LSTM")

    print(f"\nGRU  final ppl: {gru_ppls[-1]:.2f}")
    print(f"LSTM final ppl: {lstm_ppls[-1]:.2f}")
    winner = "GRU" if gru_ppls[-1] < lstm_ppls[-1] else "LSTM"
    print(f"Lower perplexity: {winner}")
