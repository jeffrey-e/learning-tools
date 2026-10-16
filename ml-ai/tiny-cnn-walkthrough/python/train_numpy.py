"""Train the tiny CNN from random weights with hand-written backpropagation in NumPy
(plain SGD). The page shows the update() function for SGD + momentum and Adam.

Generated from the Tiny CNN Walkthrough page (index.html) with its default settings:
the vertical-line image and the hand-set weights. The page's copy buttons give a
version with your own drawing, weights or training settings.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def make_dataset():
    X, y = [], []
    for w in (1, 2):                                   # vertical and horizontal lines
        for c in range(7 - w):
            v = np.zeros((6, 6)); v[:, c:c + w] = 1
            X += [v, v.T.copy()]; y += [0, 1]
    crosses = [(c, r, 2) for c in (1, 2, 3) for r in (1, 2, 3)] + [(1, 1, 1), (3, 2, 1), (4, 4, 1)]
    for c, r, w in crosses:
        m = np.zeros((6, 6)); m[:, c:c + w] = 1; m[r:r + w, :] = 1
        X.append(m); y.append(2)
    for s in (4, 5, 6):                                # square outlines
        for r in range(7 - s):
            for c in range(7 - s):
                m = np.zeros((6, 6)); m[r:r + s, c:c + s] = 1; m[r + 1:r + s - 1, c + 1:c + s - 1] = 0
                X.append(m); y.append(2)
    X += [np.zeros((6, 6))] * 6; y += [3] * 6          # empty images
    return np.array(X), np.array(y)   # 54 images, labels 0-3

def forward(img, KV, KH, W, b):
    pools, cache = [], []
    for K in (KV, KH):
        conv = (sliding_window_view(img, (3, 3)) * K).sum(axis=(2, 3))
        act = np.maximum(0, conv)                      # ReLU
        blocks = act.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)   # the four 2×2 blocks
        pools.append(blocks.max(axis=1)); cache.append((conv, blocks.argmax(axis=1)))
    x = np.concatenate(pools)
    z = W @ x + b
    p = np.exp(z - z.max()); p /= p.sum()
    return x, p, cache

def gradients(img, label, KV, KH, W, b):
    x, p, cache = forward(img, KV, KH, W, b)
    dz = p.copy(); dz[label] -= 1                      # softmax + cross-entropy: p − y
    dW, db = np.outer(dz, x), dz
    dx = W.T @ dz
    windows = sliding_window_view(img, (3, 3))
    dK = []
    for c, (conv, arg) in enumerate(cache):
        dblocks = np.zeros((4, 4))
        dblocks[np.arange(4), arg] = dx[4 * c:4 * c + 4]   # only each block's max gets gradient
        dconv = dblocks.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
        dconv *= conv > 0                                   # ReLU passes gradient only where active
        dK.append((windows * dconv[:, :, None, None]).sum(axis=(0, 1)))
    return -np.log(p[label]), dW, db, dK

X, y = make_dataset()
rng = np.random.default_rng(0)
lim_k, lim_d = np.sqrt(6 / 27), np.sqrt(6 / 12)       # Glorot uniform, as Keras uses
KV, KH = rng.uniform(-lim_k, lim_k, (2, 3, 3))
W, b = rng.uniform(-lim_d, lim_d, (4, 8)), np.zeros(4)
lr, epochs, batch = 0.3, 100, 8

def update(w, g, state):          # plain SGD
    return w - lr * g

state = {name: {} for name in ("W", "b", "KV", "KH")}   # optimizer memory per parameter
step = 0
for epoch in range(epochs):
    order = rng.permutation(len(X))
    for s in range(0, len(X), batch):
        idx = order[s:s + batch]
        g = [gradients(X[i], y[i], KV, KH, W, b) for i in idx]
        step += 1
        W = update(W, np.mean([gi[1] for gi in g], axis=0), state["W"])
        b = update(b, np.mean([gi[2] for gi in g], axis=0), state["b"])
        KV = update(KV, np.mean([gi[3][0] for gi in g], axis=0), state["KV"])
        KH = update(KH, np.mean([gi[3][1] for gi in g], axis=0), state["KH"])
    if epoch % 20 == 0 or epoch == epochs - 1:
        loss = np.mean([gradients(X[i], y[i], KV, KH, W, b)[0] for i in range(len(X))])
        acc = np.mean([forward(X[i], KV, KH, W, b)[1].argmax() == y[i] for i in range(len(X))])
        print(f"epoch {epoch:3d}  loss {loss:.3f}  accuracy {acc:.2f}")

print(KV.round(2), KH.round(2), W.round(2), b.round(2), sep="\n")
