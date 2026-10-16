"""Forward pass of the tiny CNN in NumPy: convolution, ReLU, max pooling, flatten,
dense layer and softmax on a 6x6 image.

Generated from the Tiny CNN Walkthrough page (index.html) with its default settings:
the vertical-line image and the hand-set weights. The page's copy buttons give a
version with your own drawing, weights or training settings.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

np.set_printoptions(precision=4, suppress=True)   # at most 4 decimals in printouts

img = np.array([[0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0]])
print(img.shape)

K = np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]])   # vertical kernel
windows = sliding_window_view(img, (3, 3))
print(windows.shape)   # one 3×3 patch per position

conv = (windows * K).sum(axis=(2, 3))
print(conv)

act = np.maximum(0, conv)   # ReLU
print(act)

# split rows and columns into blocks of 2, keep each block's max
pool = act.reshape(2, 2, 2, 2).max(axis=(1, 3))
print(pool)

def features(K):
    conv = (sliding_window_view(img, (3, 3)) * K).sum(axis=(2, 3))
    act = np.maximum(0, conv)
    return act.reshape(2, 2, 2, 2).max(axis=(1, 3))

KV = np.array([[1, 0, -1],
               [1, 0, -1],
               [1, 0, -1]])
KH = np.array([[1, 1, 1],
               [0, 0, 0],
               [-1, -1, -1]])

pool_v, pool_h = features(KV), features(KH)
print(pool_v.ravel(), pool_h.ravel())

x = np.concatenate([pool_v.ravel(), pool_h.ravel()])
print(x.shape)
print(x)

W = np.array([[1, 1, 1, 1, -1, -1, -1, -1],
              [-1, -1, -1, -1, 1, 1, 1, 1],
              [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
              [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]])   # (4, 8): one row per class
b = np.array([0, 0, -3, 2])

logits = W @ x + b
print(logits)

p = np.exp(logits - logits.max())   # subtract max for numerical stability
p = p / p.sum()
print(p.round(3))

CLASSES = ["Vertical line", "Horizontal line", "Cross / box", "Empty"]
print(CLASSES[p.argmax()])
