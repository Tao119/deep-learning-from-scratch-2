import sys
sys.path.append("..")
import numpy as np
from common.time_layers import TimeEmbedding, TimeRNN, TimeAffine, TimeSoftmaxWithLoss
from common.optimizer import SGD
from common.util import preprocess


class SimpleRNNLM:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        embed_W = (np.random.randn(V, D) / 100).astype("f")
        rnn_Wx  = (np.random.randn(D, H) / np.sqrt(D)).astype("f")
        rnn_Wh  = (np.random.randn(H, H) / np.sqrt(H)).astype("f")
        rnn_b   = np.zeros(H).astype("f")
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer  = self.layers[1]
        self.params, self.grads = [], []
        for l in self.layers:
            self.params += l.params
            self.grads  += l.grads

    def forward(self, xs, ts):
        for l in self.layers:
            xs = l.forward(xs)
        return self.loss_layer.forward(xs, ts)

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()


def load_ptb_mini(text=None):
    if text is None:
        text = ("the dog ran . the cat sat . the dog sat . "
                "a cat ran . a dog ran . the cat ran .")
    corpus, word_to_id, id_to_word = preprocess(text)
    return corpus, word_to_id, id_to_word


if __name__ == "__main__":
    corpus, word_to_id, id_to_word = load_ptb_mini()
    vocab_size   = len(word_to_id)
    wordvec_size = 100
    hidden_size  = 100
    time_size    = 5
    batch_size   = 4
    max_epoch    = 100
    lr           = 0.1
    max_grad     = 0.25

    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    max_iter  = max(1, data_size // (batch_size * time_size))

    model = SimpleRNNLM(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)

    time_idx = 0
    ppl_list = []
    for epoch in range(max_epoch):
        model.reset_state()
        total_loss = total_count = 0
        for _ in range(max_iter):
            batch_xs = np.zeros((batch_size, time_size), dtype=np.int32)
            batch_ts = np.zeros((batch_size, time_size), dtype=np.int32)
            data_offsets = [data_size * i // batch_size for i in range(batch_size)]
            for t in range(time_size):
                for b in range(batch_size):
                    batch_xs[b, t] = xs[(data_offsets[b] + time_idx) % data_size]
                    batch_ts[b, t] = ts[(data_offsets[b] + time_idx) % data_size]
            time_idx = (time_idx + time_size) % data_size

            loss = model.forward(batch_xs, batch_ts)
            model.backward()
            from common.util import clip_grads
            clip_grads(model.grads, max_grad)
            optimizer.update(model.params, model.grads)
            total_loss += loss
            total_count += 1

        ppl = np.exp(total_loss / total_count)
        ppl_list.append(float(ppl))
        if epoch % 20 == 0:
            print(f"epoch {epoch+1:>3}  perplexity: {ppl:.2f}")

    print(f"\nfinal perplexity: {ppl_list[-1]:.2f}")
