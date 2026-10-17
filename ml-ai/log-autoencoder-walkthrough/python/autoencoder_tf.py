# Log autoencoder walkthrough: anomaly detection in SSH auth.log with a Keras Conv1D autoencoder.
# Run from the project folder (it reads data/auth.log and python/weights_<tokenizer>.json).

import json
import re

import numpy as np
import keras
from keras import layers

lines = open("data/auth.log", encoding="utf-8").read().splitlines()
# Keep the sshd lines, and only their message: the date, host and process id change on every line
SSHD = re.compile(r"^\w{3} +\d+ [\d:]+ \S+ sshd\[\d+\]: (.*)$")
messages = [m.group(1) for line in lines if (m := SSHD.match(line))]
print(len(lines), "lines in auth.log,", len(messages), "from sshd")
print(messages[1])

# Labels by rule, first match wins. The model never sees them: they pick the training lines
# and tell us afterwards which detections were right.
RULES = [
    ("attack", r"\[preauth\]|Failed |[Ii]nvalid user|authentication failure|check pass|maximum authentication attempts|ignoring max retries|more authentication failures|Bad protocol version|Did not receive identification|message repeated|Connection reset"),
    ("legit", r"^Accepted |session opened|session closed|disconnected by user|^Disconnected from (user \S+ )?[0-9.]+ port \d+$|^Starting session"),
]

def label(msg):
    for name, rule in RULES:
        if re.search(rule, msg):
            return name
    return "other"

labels = [label(m) for m in messages]
for name in ("legit", "attack", "other"):
    print(f"{name:>6} {labels.count(name):5d} {labels.count(name) / len(labels):6.1%}")

# Train on legitimate lines only: 60% train, 20% validation (sets the threshold), 20% test.
# Every attack line is test data: the model never sees one during training.
legit = [i for i, lab in enumerate(labels) if lab == "legit"]
np.random.default_rng(1).shuffle(legit)
n = len(legit)
train_ids, val_ids, test_ids = legit[:int(0.6 * n)], legit[int(0.6 * n):int(0.8 * n)], legit[int(0.8 * n):]
attack_ids = [i for i, lab in enumerate(labels) if lab == "attack"]
other_ids = [i for i, lab in enumerate(labels) if lab == "other"]
print(f"{len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} + {len(attack_ids)} test")

weights = json.load(open("python/weights_chars.json"))   # the trained model for this tokenizer
vocab = weights["vocab"]   # every token in the training lines; 0 = padding, 1 = unknown
index = {t: i for i, t in enumerate(vocab)}
L = 128                    # every line becomes exactly L ids: cut or padded

def tokenize(msg):
    return list(msg)   # every character is a token

def encode(msg):
    ids = [index.get(t, 1) for t in tokenize(msg)[:L]]
    return ids + [0] * (L - len(ids))

line = "Received disconnect from 85.245.107.41 port 54259:11: disconnected by user"
toks = tokenize(line)[:L]
x = np.array([encode(line)])
n = len(toks)
print(len(vocab), "tokens, for example", vocab[2:10])
print(toks)
print(x[0, :n].tolist())
print(L - n, "padding positions")

def build_model(V):
    return keras.Sequential([
        keras.Input((L,), dtype="int32"),
        layers.Embedding(V, 8, name="embedding"),
        # Encoder: squeeze 128 positions into 32
        layers.Conv1D(16, 5, padding="same", activation="relu", name="enc_conv1"),
        layers.MaxPooling1D(2, name="enc_pool1"),
        layers.Conv1D(4, 5, padding="same", activation="relu", name="enc_conv2"),
        layers.MaxPooling1D(2, name="bottleneck"),
        # Decoder: expand back to 128 positions
        layers.UpSampling1D(2, name="dec_up1"),
        layers.Conv1D(4, 5, padding="same", activation="relu", name="dec_conv1"),
        layers.UpSampling1D(2, name="dec_up2"),
        layers.Conv1D(16, 5, padding="same", activation="relu", name="dec_conv2"),
        layers.Conv1D(V, 1, activation="softmax", name="output"),   # a probability per token
    ])

model = build_model(len(vocab))
model.set_weights([np.array(v, dtype="float32") for v in weights["weights"]])   # the trained weights
print(model.count_params(), "parameters")

# Run the line through the layers one by one and keep every output
outs, h = {}, x
for layer in model.layers:
    h = layer(h)
    outs[layer.name] = np.array(h)[0]
    print(f"{layer.name:>11} {str(tuple(h.shape)):>14}")

# Each id picks one learned row of 8 numbers
emb = outs["embedding"]   # (128, 8)
for t in range(5):
    print(f"{t:>3} {toks[t]!r:>12} " + " ".join(f"{v:7.4f}" for v in emb[t]))

# enc_conv1: 16 filters, each looks at 5 neighbouring tokens (positions t-2 … t+2)
print("enc_conv1 at position 0:")
print(" ".join(f"{v:.3f}" for v in outs["enc_conv1"][0]))
# The bottleneck: 32 positions × 4 numbers. Everything the decoder gets to see
z = outs["bottleneck"]
for t in range(4):
    print(f"z[{t}] " + " ".join(f"{v:7.3f}" for v in z[t]))

probs = outs["output"]   # (128, V): a probability for every token in the vocabulary, at every position
pos = 0
print("true token:", repr(toks[pos]))
for i in np.argsort(-probs[pos])[:3]:
    print(f"{vocab[i]!r:>12} {probs[pos, i]:.3f}")

# Error per token: -log(probability the model gave the true token)
p_true = probs[np.arange(n), x[0, :n]]
error = -np.log(np.maximum(p_true, 1e-7))
recon = [{0: "·", 1: "?"}.get(i, vocab[i]) for i in probs[:n].argmax(axis=1)]
score = error.mean()
threshold = 2.43
print("".join(toks))
print("".join(recon))
print(f"score {score:.4f}, threshold {threshold}:", "anomaly" if score > threshold else "normal")

# Score every line with the trained model, then compare with the threshold
X = np.array([encode(m) for m in messages])
P = model.predict(X, verbose=0, batch_size=256)
p_true = np.take_along_axis(P, X[..., None], axis=2)[..., 0]
real = X > 0
scores = (-np.log(np.maximum(p_true, 1e-7)) * real).sum(axis=1) / np.maximum(real.sum(axis=1), 1)

threshold = 2.43
flagged = scores > threshold
groups = [("validation legit", val_ids), ("test legit", test_ids), ("attack", attack_ids), ("other", other_ids)]
for name, ids in groups:
    print(f"{name:>16}: {flagged[ids].sum():4d} of {len(ids):4d} flagged")
caught, false_alarms = flagged[attack_ids].sum(), flagged[test_ids].sum()
print(f"precision {caught / max(caught + false_alarms, 1):.3f}, recall {caught / len(attack_ids):.3f}")
