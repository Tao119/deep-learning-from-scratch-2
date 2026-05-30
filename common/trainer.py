import sys
import time
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, eval_interval=20, verbose=True):
        data_size = len(x)
        max_iter = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            idx = np.random.permutation(data_size)
            x, t = x[idx], t[idx]
            for i in range(max_iter):
                xb = x[i*batch_size:(i+1)*batch_size]
                tb = t[i*batch_size:(i+1)*batch_size]
                loss = model.forward(xb, tb)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
                if (eval_interval is not None) and (i % eval_interval) == 0:
                    avg = total_loss / loss_count
                    elapsed = time.time() - start_time
                    if verbose:
                        print(f"| epoch {epoch+1} | iter {i+1}/{max_iter} | time {elapsed:.2f}s | loss {avg:.4f}")
                    self.loss_list.append(float(avg))
                    total_loss, loss_count = 0, 0
            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label="train")
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("loss")
        plt.show()


def remove_duplicate(params, grads):
    params = params[:]
    grads = grads[:]
    while True:
        find_flg = False
        L = len(params)
        for i in range(0, L - 1):
            for j in range(i + 1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                if find_flg:
                    break
            if find_flg:
                break
        if not find_flg:
            break
    return params, grads
