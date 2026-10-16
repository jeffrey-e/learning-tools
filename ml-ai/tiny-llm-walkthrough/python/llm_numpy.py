# Tiny LLM walkthrough: one forward pass of a single-head transformer, in NumPy.

import re
import numpy as np

vocab = ["the", "cat", "dog", "sat", "ran", "on", "mat", "."]
text = "the cat sat on the"
pieces = re.findall(r"[a-z]+|[^\sa-z]", text.lower())   # words and punctuation marks
words = [p for p in pieces if p in vocab]
ids = [vocab.index(p) for p in words]
print(ids)
print("not in vocabulary:", [p for p in pieces if p not in vocab])

def show(matrix, labels):
    for label, row in zip(labels, matrix):
        print(f"{label:>5} " + " ".join(f"{v:7.3f}" for v in row))

# Byte pair encoding: learn merges by counting pairs, then split the text with them.
# Real tokenizers start from the 256 byte values; letters are used here so the pieces stay readable.
corpus = ["the cat sat on the mat .", "the dog sat on the mat .", "the cat ran on the mat .", "the dog ran on the mat ."] + [text]
corpus_words = [p for line in corpus for p in re.findall(r"[a-z]+|[^sa-z]", line.lower())]

def merge(word, a, b):
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out

def bpe_train(words, num_merges):
    words = [list(w) for w in words]
    merges = []
    for _ in range(num_merges):
        counts = {}
        for w in words:
            for pair in zip(w, w[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        (a, b), count = max(counts.items(), key=lambda kv: kv[1])   # ties: the pair seen first
        merges.append((a, b, count))
        words = [merge(w, a, b) for w in words]
    return merges

def bpe_encode(word, merges):
    parts = list(word)
    for a, b, _ in merges:
        parts = merge(parts, a, b)
    return parts

merges = bpe_train(corpus_words, 10)
for k, (a, b, count) in enumerate(merges, 1):
    print(f"{k:>2}. {a} + {b} -> {a + b}  ({count}x)")
subwords = [bpe_encode(p, merges) for p in pieces]
print(" ".join("+".join(parts) for parts in subwords))
print(f"{sum(map(len, pieces))} characters -> {sum(map(len, subwords))} tokens")

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
