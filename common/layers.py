import numpy as np


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class Relu:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        self.x = x
        return x @ W + b

    def backward(self, dout):
        W, b = self.params
        dx = dout @ W.T
        dW = self.x.T @ dout
        db = dout.sum(axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = _softmax(x)
        return _cross_entropy(self.y, t)

    def backward(self, dout=1):
        batch = self.t.shape[0]
        dx = self.y.copy()
        if self.t.ndim == 1:
            dx[np.arange(batch), self.t] -= 1
        else:
            dx -= self.t
        return dx * dout / batch


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        self.x = x
        return x @ W

    def backward(self, dout):
        W, = self.params
        dx = dout @ W.T
        dW = self.x.T @ dout
        self.grads[0][...] = dW
        return dx


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy(y, t):
    if y.ndim == 1:
        y = y[np.newaxis]
        t = t[np.newaxis]
    if t.ndim == 2:
        t = t.argmax(axis=1)
    batch = y.shape[0]
    return -np.log(y[np.arange(batch), t] + 1e-7).mean()
