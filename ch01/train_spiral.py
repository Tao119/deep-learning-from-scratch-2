import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from common.layers import Affine, Sigmoid, SoftmaxWithLoss
from common.optimizer import SGD


def generate_spiral(N=100, num_class=3):
    x = np.zeros((N * num_class, 2))
    t = np.zeros(N * num_class, dtype=int)
    for j in range(num_class):
        r = np.linspace(0, 1, N)
        theta = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        x[j*N:(j+1)*N] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        t[j*N:(j+1)*N] = j
    return x, t


class TwoLayerNet:
    def __init__(self, in_size, hidden_size, out_size):
        W1 = 0.01 * np.random.randn(in_size, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = 0.01 * np.random.randn(hidden_size, out_size)
        b2 = np.zeros(out_size)
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]
        self.loss_layer = SoftmaxWithLoss()
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        return self.loss_layer.forward(score, t)

    def backward(self):
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


if __name__ == "__main__":
    x, t = generate_spiral()
    model = TwoLayerNet(2, 10, 3)
    optimizer = SGD(lr=1.0)

    max_epoch = 300
    batch_size = 30
    data_size = len(x)
    max_iter = data_size // batch_size
    loss_list = []

    for epoch in range(max_epoch):
        idx = np.random.permutation(data_size)
        x, t = x[idx], t[idx]
        total_loss = 0
        for i in range(max_iter):
            xb = x[i*batch_size:(i+1)*batch_size]
            tb = t[i*batch_size:(i+1)*batch_size]
            loss = model.forward(xb, tb)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
        avg = total_loss / max_iter
        loss_list.append(avg)
        if epoch % 50 == 0:
            print(f"epoch {epoch+1:>3}  loss={avg:.4f}")

    y = model.predict(x)
    acc = (y.argmax(axis=1) == t).mean()
    print(f"\nfinal accuracy: {acc*100:.1f}%")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    h = 0.01
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.argmax(axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    colors = ["b", "r", "g"]
    for c, label in zip(range(3), colors):
        plt.scatter(x[t == c, 0], x[t == c, 1], c=label, s=20)
    plt.title(f"Decision Boundary (acc={acc*100:.1f}%)")
    plt.tight_layout()
    plt.savefig("spiral_result.png", dpi=120)
    print("saved: spiral_result.png")
