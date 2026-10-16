# Tiny LLM walkthrough: one forward pass of a single-head transformer, in NumPy.

import numpy as np

vocab = ["the", "cat", "dog", "sat", "ran", "on", "mat", "."]
text = "the cat sat on the"
words = text.split()
ids = [vocab.index(word) for word in words]
print(ids)

def show(matrix, labels):
    for label, row in zip(labels, matrix):
        print(f"{label:>5} " + " ".join(f"{v:7.3f}" for v in row))

# Embedding table: one row per token. Columns: animal, place, action, grammar
E = np.array([
    [0.0, 0.0, 0.0, 1.0],  # the
    [1.0, 0.0, 0.1, 0.0],  # cat
    [0.9, 0.0, 0.3, 0.0],  # dog
    [0.0, 0.2, 1.0, 0.0],  # sat
    [0.1, 0.0, 1.0, 0.0],  # ran
    [0.0, 0.6, 0.0, 0.6],  # on
    [0.0, 1.0, 0.0, 0.0],  # mat
    [0.0, 0.0, 0.0, -1.0],  # .
])
show(E[ids], words)              # look up one row per token

# Cosine similarity: 1 = same direction, 0 = unrelated, -1 = opposite
unit = E / np.linalg.norm(E, axis=1, keepdims=True)
show(unit @ unit.T, vocab)

def positional(n, d):
    """Sinusoidal position encoding, as in 'Attention Is All You Need'."""
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / 10000 ** (2 * i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe

d = E.shape[1]
X = E[ids] + positional(len(ids), d)   # token meaning + position
show(X, words)

# Learned projections: query (what am I looking for?), key (what do I contain?), value (what do I pass on?)
Wq = np.array([
    [1.19, 1.91, 1.55, -0.15],
    [-0.24, -0.39, -1.05, 0.6],
    [-1.35, 0.66, -2.57, 1.53],
    [0.17, 0.89, -0.07, 0.57],
])
Wk = np.array([
    [1.26, 0.67, 2.41, -0.93],
    [1.21, 1.06, 2.31, -1.06],
    [-0.75, 2.19, -1.53, 1.26],
    [-0.74, -2.1, -1.41, -0.06],
])
Wv = np.array([
    [-4.88, -1.56, 1.52, -4.32],
    [0.28, -5.4, 4.84, 2.69],
    [-3.78, 0.38, -3.49, 1.16],
    [0.02, 0.23, -1.79, 1.52],
])
Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores = Q @ K.T / np.sqrt(d)

# Causal mask: each word may only look at itself and earlier words
n = len(ids)
scores = np.where(np.tril(np.ones((n, n))) == 1, scores, -np.inf)
A = np.exp(scores - scores.max(axis=1, keepdims=True))
A = A / A.sum(axis=1, keepdims=True)   # softmax per row
show(A, words)

# Learned output projection
Wo = np.array([
    [-2.19, -2.13, -2.29, -4.7],
    [-2.6, 3.45, -4.33, -0.39],
    [-1.89, -3.42, 5.26, 0.27],
    [2.86, -3.73, -2.49, -3.19],
])
H = X + (A @ V) @ Wo       # residual: add what attention found to each position
logits = H[-1] @ E.T       # compare the last position with every token embedding

temperature = 1.0
p = np.exp((logits - logits.max()) / temperature)
p = p / p.sum()
for word, prob in sorted(zip(vocab, p), key=lambda t: -t[1]):
    print(f"{word:>5} {prob:.3f}")
