import sys
sys.path.append("..")
import numpy as np
from common.time_layers import TimeLSTM, TimeEmbedding, TimeSoftmaxWithLoss
from common.layers import Affine
from common.optimizer import Adam
from common.util import preprocess, clip_grads
from common.time_layers import TimeAffine


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape
        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = t.sum(axis=1)
        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        dt = dc[:, np.newaxis, :].repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = dar.sum(axis=2)
        return dhs, da


class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = _softmax
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape
        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        t = hs * hr
        s = t.sum(axis=2)
        a = self.softmax(s)
        self.cache = (hs, hr, a)
        return a

    def backward(self, da):
        hs, hr, a = self.cache
        N, T, H = hs.shape
        ds = da * a * (1 - a)
        dt = ds[:, :, np.newaxis].repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = dhr.sum(axis=1)
        return dhs, dh


class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        self.attention_weight = a
        return self.weight_sum_layer.forward(hs, a)

    def backward(self, dc):
        dhs0, da = self.weight_sum_layer.backward(dc)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh


class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []
        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)
        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)
        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh
        return dhs_enc, dhs_dec


class AttentionSeq2Seq:
    """Seq2Seq with Attention for sequence reversal task"""

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        def lstm_params(in_h, out_h):
            Wx = (np.random.randn(in_h, 4*out_h) / np.sqrt(in_h)).astype("f")
            Wh = (np.random.randn(out_h, 4*out_h) / np.sqrt(out_h)).astype("f")
            b  = np.zeros(4*out_h).astype("f")
            return Wx, Wh, b

        embed_W_enc = (np.random.randn(V, D) / 100).astype("f")
        embed_W_dec = (np.random.randn(V, D) / 100).astype("f")
        affine_W = (np.random.randn(2*H, V) / np.sqrt(2*H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.encoder_embed = TimeEmbedding(embed_W_enc)
        self.encoder_lstm  = TimeLSTM(*lstm_params(D, H), stateful=False)
        self.decoder_embed = TimeEmbedding(embed_W_dec)
        self.decoder_lstm  = TimeLSTM(*lstm_params(D, H), stateful=True)
        self.attention     = TimeAttention()
        self.affine        = TimeAffine(affine_W, affine_b)
        self.loss_layer    = TimeSoftmaxWithLoss()

        self.affine = TimeAffine(affine_W, affine_b)

        layers = [self.encoder_embed, self.encoder_lstm,
                  self.decoder_embed, self.decoder_lstm,
                  self.affine]
        self.params = []
        self.grads  = []
        for l in layers:
            self.params += l.params
            self.grads  += l.grads

    def forward(self, xs, ts):
        dec_xs, dec_ts = ts[:, :-1], ts[:, 1:]
        hs_enc = self.encoder_embed.forward(xs)
        hs_enc = self.encoder_lstm.forward(hs_enc)

        h, c = self.encoder_lstm.h, self.encoder_lstm.c
        self.decoder_lstm.set_state(h, c)

        hs_dec = self.decoder_embed.forward(dec_xs)
        hs_dec = self.decoder_lstm.forward(hs_dec)

        c_dec = self.attention.forward(hs_enc, hs_dec)
        out = np.concatenate([c_dec, hs_dec], axis=2)
        score = self.affine.forward(out)
        return self.loss_layer.forward(score, dec_ts)

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dout = self.affine.backward(dout)
        N, T, _ = dout.shape
        H = dout.shape[2] // 2
        dc_dec = dout[:, :, :H]
        dhs_dec = dout[:, :, H:]

        dhs_enc, dhs_dec2 = self.attention.backward(dc_dec)
        dhs_dec = dhs_dec + dhs_dec2

        dout = self.decoder_lstm.backward(dhs_dec)
        self.decoder_embed.backward(dout)

        dxs_enc = self.encoder_lstm.backward(dhs_enc)
        self.encoder_embed.backward(dxs_enc)

    def generate(self, xs, start_id, sample_size):
        hs_enc = self.encoder_embed.forward(xs)
        hs_enc = self.encoder_lstm.forward(hs_enc)
        h, c = self.encoder_lstm.h, self.encoder_lstm.c
        self.decoder_lstm.set_state(h, c)

        sampled = []
        char = start_id
        for _ in range(sample_size):
            x = np.array([[char]])
            h_emb = self.decoder_embed.forward(x)
            h_dec = self.decoder_lstm.forward(h_emb)
            c_dec = self.attention.forward(hs_enc, h_dec)
            out = np.concatenate([c_dec, h_dec], axis=2)
            score = self.affine.forward(out)
            char = score[0, 0].argmax()
            sampled.append(int(char))
        return sampled


def make_reverse_dataset(n=1000, seq_len=5, vocab_size=10):
    xs = np.random.randint(1, vocab_size, (n, seq_len))
    ts = np.zeros((n, seq_len + 1), dtype=int)
    ts[:, 0] = 0
    ts[:, 1:] = xs[:, ::-1]
    return xs, ts


if __name__ == "__main__":
    n_train, n_test = 9000, 1000
    xs, ts = make_reverse_dataset(n_train + n_test, seq_len=5, vocab_size=13)
    x_train, t_train = xs[:n_train], ts[:n_train]
    x_test,  t_test  = xs[n_train:], ts[n_train:]

    vocab_size = 13
    wordvec_size = 16
    hidden_size  = 128
    batch_size   = 128
    max_epoch    = 30

    model = AttentionSeq2Seq(vocab_size, wordvec_size, hidden_size)
    optimizer = Adam(lr=0.001)

    data_size = len(x_train)
    max_iter  = data_size // batch_size

    for epoch in range(max_epoch):
        idx = np.random.permutation(data_size)
        x_train = x_train[idx]
        t_train = t_train[idx]
        total = cnt = 0
        for i in range(max_iter):
            xb = x_train[i*batch_size:(i+1)*batch_size]
            tb = t_train[i*batch_size:(i+1)*batch_size]
            loss = model.forward(xb, tb)
            model.backward()
            clip_grads(model.grads, 5.0)
            optimizer.update(model.params, model.grads)
            total += loss; cnt += 1

        acc = 0
        for i in range(0, len(x_test), batch_size):
            xb = x_test[i:i+batch_size]
            tb = t_test[i:i+batch_size]
            for j, x in enumerate(xb):
                model.encoder_lstm.reset_state()
                gen = model.generate(x[np.newaxis], start_id=0, sample_size=5)
                correct = tb[j, 1:]
                acc += int(np.all(np.array(gen) == correct))
        acc /= len(x_test)
        print(f"epoch {epoch+1:>2}  loss={total/cnt:.3f}  acc={acc*100:.1f}%")
