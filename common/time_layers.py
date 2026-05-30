import numpy as np
from common.layers import Affine, SoftmaxWithLoss
from common.embedding import Embedding


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads  = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache  = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = x @ Wx + h_prev @ Wh + b
        h_next = np.tanh(t)
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache
        dt = dh_next * (1 - h_next**2)
        db = dt.sum(axis=0)
        dWh = h_prev.T @ dt
        dh_prev = dt @ Wh.T
        dWx = x.T @ dt
        dx  = dt @ Wx.T
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads  = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape
        self.layers = []
        hs = np.empty((N, T, H), dtype="f")
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        for t in range(T):
            layer = RNN(*[p.copy() for p in self.params])
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]
        dxs = np.empty((N, T, D), dtype="f")
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx
            for i, g in enumerate(layer.grads):
                grads[i] += g
        for i, g in enumerate(grads):
            self.grads[i][...] = g
        self.dh = dh
        return dxs


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads  = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache  = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        A = x @ Wx + h_prev @ Wh + b
        f = _sigmoid(A[:, :H])
        g = np.tanh(A[:, H:2*H])
        i = _sigmoid(A[:, 2*H:3*H])
        o = _sigmoid(A[:, 3*H:])
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache
        tanh_c = np.tanh(c_next)
        ds = dc_next + (dh_next * o) * (1 - tanh_c**2)
        dc_prev = ds * f
        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c
        dg = ds * i
        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g**2)
        dA = np.hstack([df, dg, di, do])
        dWh = h_prev.T @ dA
        dWx = x.T @ dA
        db  = dA.sum(axis=0)
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        dx = dA @ Wx.T
        dh_prev = dA @ Wh.T
        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads  = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        self.layers = []
        hs = np.empty((N, T, H), dtype="f")
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype="f")
        for t in range(T):
            layer = LSTM(*[p.copy() for p in self.params])
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]
        dxs = np.empty((N, T, D), dtype="f")
        dh, dc = 0, 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, g in enumerate(layer.grads):
                grads[i] += g
        for i, g in enumerate(grads):
            self.grads[i][...] = g
        self.dh = dh
        return dxs


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads  = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape
        out = np.empty((N, T, D), dtype="f")
        self.layers = []
        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        return out

    def backward(self, dout):
        N, T, D = dout.shape
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]
        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads  = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        self.x = x
        rx = x.reshape(N*T, D)
        out = rx @ self.params[0] + self.params[1]
        return out.reshape(N, T, -1)

    def backward(self, dout):
        N, T, D = self.x.shape
        W, b = self.params
        dout = dout.reshape(N*T, -1)
        rx = self.x.reshape(N*T, D)
        db = dout.sum(axis=0)
        dW = rx.T @ dout
        dx = dout @ W.T
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx.reshape(N, T, D)


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape
        if ts.ndim == 3:
            ts = ts.argmax(axis=2)
        mask = ts != self.ignore_label
        xs = xs.reshape(N*T, V)
        ts = ts.reshape(N*T)
        mask = mask.reshape(N*T)
        ys = _softmax(xs)
        ls = np.log(ys[np.arange(N*T), ts] + 1e-7)
        ls *= mask
        loss = -ls.sum() / mask.sum()
        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache
        dx = ys.copy()
        dx[np.arange(N*T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]
        return dx.reshape((N, T, V))


def _sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
