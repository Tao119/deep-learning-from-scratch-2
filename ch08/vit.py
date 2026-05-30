"""
Vision Transformer (ViT) — tiny pure NumPy implementation.

Dataset
-------
500 synthetic 8×8 grayscale images:
  class 0: circle-like pattern (concentric, bright centre)
  class 1: square-like pattern (bright rectangular border)

Architecture
------------
  patch size 2×2  →  16 patches per image
  Patch embedding : flatten(4) → Affine(4→32)
  Class token     : learnable 32-dim vector prepended  →  seq len = 17
  Positional enc  : learnable (17×32)
  Transformer Enc : 2 blocks, 2 heads, d_model=32, d_ff=64
  Class head      : LayerNorm → Affine(32→2)  on class token only

Outputs
-------
  Prints training accuracy vs 2-layer FC baseline.
  Saves  vit_results.png  (loss curves + attention map at patch positions).
"""

import sys
sys.path.append("..")
import numpy as np
import os


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_circle(size=8):
    img = np.zeros((size, size), dtype="f")
    cx  = cy = size / 2.0 - 0.5
    r   = size / 2.0 - 1.0
    for i in range(size):
        for j in range(size):
            d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
            img[i, j] = max(0.0, 1.0 - d / r)
    return img


def make_square(size=8, border=2):
    img = np.zeros((size, size), dtype="f")
    img[:border, :]   = 1.0
    img[-border:, :]  = 1.0
    img[:, :border]   = 1.0
    img[:, -border:]  = 1.0
    return img


def make_dataset(n=500, size=8, noise=0.15, seed=1):
    rng    = np.random.default_rng(seed)
    half   = n // 2
    circle = make_circle(size)
    square = make_square(size)
    imgs   = []
    labels = []
    for _ in range(half):
        img = circle + rng.normal(0, noise, (size, size)).astype("f")
        imgs.append(np.clip(img, 0, 1))
        labels.append(0)
    for _ in range(n - half):
        img = square + rng.normal(0, noise, (size, size)).astype("f")
        imgs.append(np.clip(img, 0, 1))
        labels.append(1)
    xs = np.stack(imgs)               # (N, 8, 8)
    ys = np.array(labels, dtype=np.int32)
    perm = rng.permutation(n)
    return xs[perm], ys[perm]


def patchify(imgs, patch_size=2):
    """
    imgs: (N, H, W)
    Returns: (N, num_patches, patch_size*patch_size)
    """
    N, H, W = imgs.shape
    P = patch_size
    n_h, n_w = H // P, W // P
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            p = imgs[:, i*P:(i+1)*P, j*P:(j+1)*P]   # (N, P, P)
            patches.append(p.reshape(N, P * P))
    return np.stack(patches, axis=1)   # (N, n_patches, P*P)


# ---------------------------------------------------------------------------
# Layer helpers
# ---------------------------------------------------------------------------

def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class Affine:
    def __init__(self, in_dim, out_dim, scale=None):
        if scale is None:
            scale = np.sqrt(in_dim)
        self.W      = (np.random.randn(in_dim, out_dim) / scale).astype("f")
        self.b      = np.zeros(out_dim, dtype="f")
        self.params = [self.W, self.b]
        self.grads  = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.cache  = None

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        x = self.cache
        orig_shape = x.shape
        xr   = x.reshape(-1, orig_shape[-1])
        dr   = dout.reshape(-1, dout.shape[-1])
        dW   = xr.T @ dr
        db   = dr.sum(axis=0)
        dx   = (dr @ self.W.T).reshape(orig_shape)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class LayerNorm:
    def __init__(self, d, eps=1e-6):
        self.eps    = eps
        self.gamma  = np.ones(d, dtype="f")
        self.beta   = np.zeros(d, dtype="f")
        self.params = [self.gamma, self.beta]
        self.grads  = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]
        self.cache  = None

    def forward(self, x):
        mu   = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1,  keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        out  = self.gamma * xhat + self.beta
        self.cache = (x, xhat, mu, var)
        return out

    def backward(self, dout):
        x, xhat, mu, var = self.cache
        D       = x.shape[-1]
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dgamma  = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
        dbeta   = dout.sum(axis=tuple(range(dout.ndim - 1)))
        dxhat   = dout * self.gamma
        dx      = std_inv * (dxhat
                             - dxhat.mean(axis=-1, keepdims=True)
                             - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta
        return dx


class MultiHeadSelfAttention:
    """
    Non-causal (bidirectional) multi-head self-attention for ViT encoder.
    Stores attention weights of the first head for visualisation.
    """

    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        scale         = np.sqrt(d_model)
        self.Wq = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.Wk = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.Wv = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.Wo = (np.random.randn(d_model, d_model) / scale).astype("f")
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.grads  = [np.zeros_like(p) for p in self.params]
        self.attn_weights = None   # saved for visualisation
        self.cache  = None

    def _split(self, x):
        N, T, _ = x.shape
        return x.reshape(N, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge(self, x):
        N, h, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, T, h * dh)

    def forward(self, x):
        N, T, _ = x.shape
        Q = self._split(x @ self.Wq)
        K = self._split(x @ self.Wk)
        V = self._split(x @ self.Wv)
        scale  = np.sqrt(self.d_head)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale   # (N, h, T, T)
        A      = _softmax(scores)
        self.attn_weights = A
        out    = self._merge(A @ V) @ self.Wo
        self.cache = (x, Q, K, V, A, scores)
        return out

    def backward(self, dout):
        x, Q, K, V, A, scores = self.cache
        N, T, _ = x.shape
        scale   = np.sqrt(self.d_head)

        out_pre = self._merge(A @ V)
        dWo     = out_pre.reshape(N * T, self.d_model).T @ dout.reshape(N * T, self.d_model)
        d_merged = dout @ self.Wo.T

        d_AV = self._split(d_merged)
        dA   = d_AV @ V.transpose(0, 1, 3, 2)
        dV   = A.transpose(0, 1, 3, 2) @ d_AV

        dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dscores /= scale

        dQ = dscores @ K
        dK = dscores.transpose(0, 1, 3, 2) @ Q

        dQ_m = self._merge(dQ)
        dK_m = self._merge(dK)
        dV_m = self._merge(dV)

        dWq = x.reshape(N*T, self.d_model).T @ dQ_m.reshape(N*T, self.d_model)
        dWk = x.reshape(N*T, self.d_model).T @ dK_m.reshape(N*T, self.d_model)
        dWv = x.reshape(N*T, self.d_model).T @ dV_m.reshape(N*T, self.d_model)
        dx  = dQ_m @ self.Wq.T + dK_m @ self.Wk.T + dV_m @ self.Wv.T

        self.grads[0][...] = dWq
        self.grads[1][...] = dWk
        self.grads[2][...] = dWv
        self.grads[3][...] = dWo
        return dx


class FFN:
    def __init__(self, d_model, d_ff):
        scale = np.sqrt(d_model)
        self.W1 = (np.random.randn(d_model, d_ff)   / scale).astype("f")
        self.b1 = np.zeros(d_ff, dtype="f")
        self.W2 = (np.random.randn(d_ff,   d_model) / np.sqrt(d_ff)).astype("f")
        self.b2 = np.zeros(d_model, dtype="f")
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.grads  = [np.zeros_like(p) for p in self.params]
        self.cache  = None

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        xr, hr, dr = x.reshape(-1, x.shape[-1]), h.reshape(-1, h.shape[-1]), dout.reshape(-1, dout.shape[-1])
        dW2 = hr.T @ dr
        db2 = dr.sum(axis=0)
        dh  = dr @ self.W2.T
        dh[hr == 0] = 0
        dW1 = xr.T @ dh
        db1 = dh.sum(axis=0)
        dx  = (dh @ self.W1.T).reshape(x.shape)
        self.grads[0][...] = dW1
        self.grads[1][...] = db1
        self.grads[2][...] = dW2
        self.grads[3][...] = db2
        return dx


class TransformerEncoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.norm1 = LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.ffn   = FFN(d_model, d_ff)
        self.params = (self.norm1.params + self.attn.params
                       + self.norm2.params + self.ffn.params)
        self.grads  = (self.norm1.grads  + self.attn.grads
                       + self.norm2.grads  + self.ffn.grads)
        self.cache  = None

    def forward(self, x):
        h   = x + self.attn.forward(self.norm1.forward(x))
        out = h + self.ffn.forward(self.norm2.forward(h))
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        dh_ffn = self.ffn.backward(self.norm2.backward(dout))
        dh     = dout + dh_ffn
        dx_attn = self.attn.backward(self.norm1.backward(dh))
        dx = dh + dx_attn
        return dx


# ---------------------------------------------------------------------------
# ViT model
# ---------------------------------------------------------------------------

class ViT:
    def __init__(self, patch_dim=4, d_model=32, n_heads=2,
                 n_layers=2, d_ff=64, n_patches=16, n_classes=2):
        self.d_model  = d_model
        self.n_patches = n_patches

        # Patch embedding
        self.patch_emb = Affine(patch_dim, d_model)

        # Class token (learnable)
        self.cls_token = np.zeros((1, 1, d_model), dtype="f")

        # Positional encoding (learnable), shape (1, 1+n_patches, d_model)
        self.pos_enc = (np.random.randn(1, 1 + n_patches, d_model) * 0.02).astype("f")

        # Transformer encoder blocks
        self.blocks = [TransformerEncoderBlock(d_model, n_heads, d_ff)
                       for _ in range(n_layers)]

        # Classification head
        self.norm_head  = LayerNorm(d_model)
        self.head       = Affine(d_model, n_classes)

        # Separate gradient tensors for learnable positional buffers
        self.d_cls_token = np.zeros_like(self.cls_token)
        self.d_pos_enc   = np.zeros_like(self.pos_enc)

        # Collect params/grads — block params come first for easy Adam indexing
        self.params = (self.patch_emb.params
                       + [self.cls_token, self.pos_enc]
                       + self.norm_head.params
                       + self.head.params)
        self.grads  = (self.patch_emb.grads
                       + [self.d_cls_token, self.d_pos_enc]
                       + self.norm_head.grads
                       + self.head.grads)
        for blk in self.blocks:
            self.params += blk.params
            self.grads  += blk.grads

        self.cache = None

    def _forward_encode(self, patches):
        """patches: (N, n_patches, patch_dim)"""
        N = patches.shape[0]
        patch_emb = self.patch_emb.forward(patches)        # (N, n_patches, d_model)
        cls = np.repeat(self.cls_token, N, axis=0)          # (N, 1, d_model)
        x   = np.concatenate([cls, patch_emb], axis=1)     # (N, 1+n_patches, d_model)
        x   = x + self.pos_enc                              # broadcast (1,T,D) → (N,T,D)
        self._x_before_blocks = x                           # save for backward
        for blk in self.blocks:
            x = blk.forward(x)
        return x

    def forward(self, patches, labels):
        """
        patches: (N, n_patches, patch_dim)
        labels : (N,) int
        Returns scalar loss.
        """
        N = patches.shape[0]
        x = self._forward_encode(patches)

        # Use class token position (index 0)
        cls_out  = x[:, 0, :]                            # (N, d_model)
        normed   = self.norm_head.forward(cls_out)
        logits   = self.head.forward(normed)              # (N, n_classes)

        # Cross-entropy
        logits_max = logits.max(axis=1, keepdims=True)
        exp_l    = np.exp(logits - logits_max)
        probs    = exp_l / exp_l.sum(axis=1, keepdims=True)
        loss     = -np.log(probs[np.arange(N), labels] + 1e-7).mean()

        self.cache = (x, cls_out, normed, logits, probs, labels, N)
        return loss

    def backward(self):
        x, cls_out, normed, logits, probs, labels, N = self.cache

        # CE gradient w.r.t. logits
        d_logits = probs.copy()
        d_logits[np.arange(N), labels] -= 1
        d_logits /= N

        # Classification head → LayerNorm → cls token
        d_normed = self.head.backward(d_logits)
        d_cls    = self.norm_head.backward(d_normed)   # (N, d_model)

        # Scatter cls gradient into full-sequence gradient tensor
        dx = np.zeros((N, 1 + self.n_patches, self.d_model), dtype="f")
        dx[:, 0, :] = d_cls

        # Backward through Transformer encoder blocks (reverse order)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)

        # Gradient w.r.t. positional encoding (broadcast across batch)
        self.d_pos_enc[...] = dx.sum(axis=0, keepdims=True)   # (1, T, d_model)

        # Gradient w.r.t. class token (sum across batch → (1,1,d_model))
        self.d_cls_token[...] = dx[:, 0:1, :].sum(axis=0, keepdims=True)

        # Gradient w.r.t. patch embeddings (positions 1:)
        d_patch_emb = dx[:, 1:, :]               # (N, n_patches, d_model)
        self.patch_emb.backward(d_patch_emb)

    def predict(self, patches):
        N = patches.shape[0]
        x = self._forward_encode(patches)
        cls_out = x[:, 0, :]
        normed  = self.norm_head.forward(cls_out)
        logits  = self.head.forward(normed)
        return logits.argmax(axis=1)

    def get_attn_weights(self, patches):
        """Return attention weights of first head, first block, for one sample."""
        self._forward_encode(patches)
        return self.blocks[0].attn.attn_weights   # (N, n_heads, T, T)


# ---------------------------------------------------------------------------
# 2-layer FC baseline
# ---------------------------------------------------------------------------

class FC2Baseline:
    def __init__(self, in_dim, hidden=64, n_classes=2):
        scale = np.sqrt(in_dim)
        self.W1 = (np.random.randn(in_dim, hidden) / scale).astype("f")
        self.b1 = np.zeros(hidden, dtype="f")
        self.W2 = (np.random.randn(hidden, n_classes) / np.sqrt(hidden)).astype("f")
        self.b2 = np.zeros(n_classes, dtype="f")
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.grads  = [np.zeros_like(p) for p in self.params]
        self.cache  = None

    def forward(self, x_flat, labels):
        N = x_flat.shape[0]
        h     = np.maximum(0, x_flat @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        lmax   = logits.max(axis=1, keepdims=True)
        exp_l  = np.exp(logits - lmax)
        probs  = exp_l / exp_l.sum(axis=1, keepdims=True)
        loss   = -np.log(probs[np.arange(N), labels] + 1e-7).mean()
        self.cache = (x_flat, h, probs, labels, N)
        return loss

    def backward(self):
        x_flat, h, probs, labels, N = self.cache
        dlogits = probs.copy()
        dlogits[np.arange(N), labels] -= 1
        dlogits /= N
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh  = dlogits @ self.W2.T
        dh[h == 0] = 0
        dW1 = x_flat.T @ dh
        db1 = dh.sum(axis=0)
        self.grads[0][...] = dW1
        self.grads[1][...] = db1
        self.grads[2][...] = dW2
        self.grads[3][...] = db2

    def predict(self, x_flat):
        h      = np.maximum(0, x_flat @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits.argmax(axis=1)


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr_t   = self.lr * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g ** 2
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_model(model, is_vit, xs_patch, xs_flat, ys,
                n_epochs=200, batch_size=32, lr=1e-3):
    optimizer  = Adam(lr=lr)
    loss_list  = []
    N = len(ys)
    for epoch in range(n_epochs):
        idx   = np.random.permutation(N)
        total = cnt = 0
        for start in range(0, N, batch_size):
            b_idx = idx[start:start + batch_size]
            if is_vit:
                loss = model.forward(xs_patch[b_idx], ys[b_idx])
                model.backward()
            else:
                loss = model.forward(xs_flat[b_idx], ys[b_idx])
                model.backward()
            # Gradient clipping
            norm = np.sqrt(sum((g ** 2).sum() for g in model.grads))
            if norm > 5.0:
                for g in model.grads:
                    g *= 5.0 / (norm + 1e-8)
            optimizer.update(model.params, model.grads)
            total += loss
            cnt   += 1
        loss_list.append(total / cnt)
    return loss_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Data ----
    imgs, labels = make_dataset(n=500)
    patches  = patchify(imgs, patch_size=2)              # (500, 16, 4)
    flat     = imgs.reshape(len(imgs), -1).astype("f")   # (500, 64)

    split = 400
    xs_patch_tr, xs_patch_te = patches[:split], patches[split:]
    xs_flat_tr,  xs_flat_te  = flat[:split],    flat[split:]
    ys_tr, ys_te             = labels[:split],  labels[split:]

    n_epochs   = 200
    batch_size = 32

    # ---- ViT ----
    print("=" * 55)
    print("Training ViT …")
    vit   = ViT(patch_dim=4, d_model=32, n_heads=2, n_layers=2, d_ff=64,
                n_patches=16, n_classes=2)
    vit_losses = train_model(vit, True, xs_patch_tr, xs_flat_tr, ys_tr,
                             n_epochs=n_epochs, batch_size=batch_size, lr=5e-4)
    vit_acc_tr = (vit.predict(xs_patch_tr) == ys_tr).mean()
    vit_acc_te = (vit.predict(xs_patch_te) == ys_te).mean()
    print(f"ViT   train_acc={vit_acc_tr*100:.1f}%  test_acc={vit_acc_te*100:.1f}%")

    # ---- FC baseline ----
    print("Training FC baseline …")
    fc   = FC2Baseline(in_dim=64, hidden=64, n_classes=2)
    fc_losses = train_model(fc, False, xs_patch_tr, xs_flat_tr, ys_tr,
                            n_epochs=n_epochs, batch_size=batch_size, lr=5e-4)
    fc_acc_tr = (fc.predict(xs_flat_tr) == ys_tr).mean()
    fc_acc_te = (fc.predict(xs_flat_te) == ys_te).mean()
    print(f"FC    train_acc={fc_acc_tr*100:.1f}%  test_acc={fc_acc_te*100:.1f}%")

    # ---- Save figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Attention map from first head, block 0, for first test image
        sample = xs_patch_te[:1]    # (1, 16, 4)
        attn   = vit.get_attn_weights(sample)   # (1, n_heads, T, T)
        attn_h0 = attn[0, 0]                     # (17, 17)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Loss curves
        ax = axes[0]
        ax.plot(vit_losses, label="ViT")
        ax.plot(fc_losses,  label="FC baseline", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()

        # Attention map (patch×patch portion, ignoring cls→cls)
        # rows = query positions, cols = key positions (17×17)
        ax = axes[1]
        im = ax.imshow(attn_h0, cmap="viridis", aspect="auto")
        ax.set_title("Attention Map\n(head 0, block 0, sample 0)")
        ax.set_xlabel("Key position (0=CLS, 1-16=patches)")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax)

        # CLS-token attention over patches (row 0, cols 1:)
        ax = axes[2]
        cls_attn = attn_h0[0, 1:].reshape(4, 4)   # 16 patches → 4×4 grid
        im2 = ax.imshow(cls_attn, cmap="hot", aspect="auto")
        ax.set_title("CLS→patch attention\n(4×4 patch grid)")
        plt.colorbar(im2, ax=ax)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), "vit_results.png")
        plt.savefig(out_path, dpi=120)
        print(f"\nFigure saved → {out_path}")
    except ImportError:
        print("\nmatplotlib not available — skipping figure.")

    print("\n=== Summary ===")
    print(f"  ViT   test accuracy : {vit_acc_te*100:.1f}%")
    print(f"  FC    test accuracy : {fc_acc_te*100:.1f}%")
