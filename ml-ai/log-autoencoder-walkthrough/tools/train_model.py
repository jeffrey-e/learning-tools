"""Trains the page's autoencoders with Keras, one per tokenizer, and writes weights.js,
python/weights_<tokenizer>.json and data.js.

Steps: keep the sshd lines of data/auth.log, strip the timestamp/host/pid prefix, label every
message with the rules below, split the legitimate messages into train/validation/test, and for
each tokenizer: build the vocabulary from the training messages, train a Conv1D autoencoder on
them, round the weights to 4 decimals and score every message with the rounded weights (so the
page, the Python and data.js agree). The split is the same for every tokenizer.

Usage:  python tools/train_model.py        (seeded and deterministic on CPU)
"""
import json
import os
import pathlib
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Same rules, in the same order, as RULES in model.js (the tests check that the labels agree).
RULES = [
    ("attack", r"\[preauth\]|Failed |[Ii]nvalid user|authentication failure|check pass|maximum authentication attempts|ignoring max retries|more authentication failures|Bad protocol version|Did not receive identification|message repeated|Connection reset"),
    ("legit", r"^Accepted |session opened|session closed|disconnected by user|^Disconnected from (user \S+ )?[0-9.]+ port \d+$|^Starting session"),
]
SSHD = re.compile(r"^\w{3} +\d+ [\d:]+ \S+ sshd\[\d+\]: (.*)$")
PROGRAM = re.compile(r"^\w{3} +\d+ [\d:]+ \S+ ([^\[:]+)")
EMBED, F1, F2, K = 8, 16, 4, 5
EPOCHS, BATCH, SEED = 150, 32, 0
SPLIT_SEED = 1
BPE_MERGES = 200


# ---- Tokenizers: same code as the Python cells model.js writes, and the same rules as its JavaScript ----
def placeholder(p):
    """A number becomes <num>, an IP address <ip>; anything else is kept."""
    return ("<ip>" if "." in p else "<num>") if p[0] in "0123456789" else p


def tok_chars(msg):
    return list(msg)


def tok_masked(msg):
    return [placeholder(p) for p in re.findall(r"[0-9]+(?:\.[0-9]+){3}|[0-9]+|[^\n]", msg)]


def tok_words(msg):
    return [placeholder(p) for p in re.findall(r"[0-9]+(?:\.[0-9]+){3}|[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^ ]", msg)]


BPE_PIECE = r"[A-Za-z_]+|[0-9]+| |[^\n]"


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


def bpe_train(messages, num_merges):
    """Learns merges from the pieces of the training messages, most frequent pair first (ties: seen first)."""
    freq = {}
    for msg in messages:
        for p in re.findall(BPE_PIECE, msg):
            freq[p] = freq.get(p, 0) + 1
    split = {p: list(p) for p in freq}
    merges = []
    for _ in range(num_merges):
        counts = {}
        for p, w in split.items():
            for pair in zip(w, w[1:]):
                counts[pair] = counts.get(pair, 0) + freq[p]
        if not counts:
            break
        (a, b), _count = max(counts.items(), key=lambda kv: kv[1])
        merges.append([a, b])
        split = {p: merge(w, a, b) for p, w in split.items()}
    return merges


def make_tok_bpe(merges):
    cache = {}

    def tok_bpe(msg):
        out = []
        for p in re.findall(BPE_PIECE, msg):
            if p not in cache:
                parts = list(p)
                for a, b in merges:
                    parts = merge(parts, a, b)
                cache[p] = parts
            out += cache[p]
        return out
    return tok_bpe


# name: (sequence length, tokenizer). The order is the order of the scores in data.js.
TOKENIZERS = {"chars": 128, "masked": 112, "words": 32, "bpe": 64}


def label(msg):
    for name, rule in RULES:
        if re.search(rule, msg):
            return name
    return "other"


def split(labels):
    """Legitimate messages: 60% train, 20% validation, 20% test. Every attack is test data."""
    legit = [i for i, lab in enumerate(labels) if lab == "legit"]
    np.random.default_rng(SPLIT_SEED).shuffle(legit)
    n = len(legit)
    return legit[: int(0.6 * n)], legit[int(0.6 * n): int(0.8 * n)], legit[int(0.8 * n):]


def build_model(V, L):
    """Same layers, names and order as model.js and the page's Python cells."""
    return keras.Sequential([
        keras.Input((L,), dtype="int32"),
        layers.Embedding(V, EMBED, name="embedding"),
        layers.Conv1D(F1, K, padding="same", activation="relu", name="enc_conv1"),
        layers.MaxPooling1D(2, name="enc_pool1"),
        layers.Conv1D(F2, K, padding="same", activation="relu", name="enc_conv2"),
        layers.MaxPooling1D(2, name="bottleneck"),
        layers.UpSampling1D(2, name="dec_up1"),
        layers.Conv1D(F2, K, padding="same", activation="relu", name="dec_conv1"),
        layers.UpSampling1D(2, name="dec_up2"),
        layers.Conv1D(F1, K, padding="same", activation="relu", name="dec_conv2"),
        layers.Conv1D(V, 1, activation="softmax", name="output"),
    ])


def scores(model, X):
    """Mean of -log p(true token) over the real (non-padding) tokens of each line."""
    P = model.predict(X, verbose=0, batch_size=256)
    p = np.take_along_axis(P, X[..., None], axis=2)[..., 0]
    nll = -np.log(np.maximum(p, 1e-7))
    mask = X > 0
    return (nll * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)


def train_one(name, L, tokenize, messages, labels, train, val, test, extra):
    toks = [tokenize(m) for m in messages]
    vocab = ["<pad>", "<unk>"] + sorted(set(t for i in train for t in toks[i]))
    index = {c: i for i, c in enumerate(vocab)}

    def encode(ts):
        ids = [index.get(t, 1) for t in ts[:L]]
        return ids + [0] * (L - len(ids))

    X = np.array([encode(t) for t in toks], dtype="int32")
    mask = (X > 0).astype("float32")

    keras.utils.set_random_seed(SEED)
    model = build_model(len(vocab), L)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    hist = model.fit(X[train], X[train], sample_weight=mask[train],
                     validation_data=(X[val], X[val], mask[val]),
                     epochs=EPOCHS, batch_size=BATCH, shuffle=True, verbose=0)

    weights = [np.round(w.astype(np.float64), 4) for w in model.get_weights()]
    model.set_weights([w.astype("float32") for w in weights])
    s = scores(model, X)
    # 3 significant digits (at least 0.0001, so Python and JavaScript print it the same way)
    threshold = max(float(f"{np.percentile(s[val], 99):.3g}"), 0.0001)

    print(f"== {name}: {len(vocab)} tokens, L={L}, threshold {threshold}, "
          f"loss {hist.history['loss'][-1]:.4f}, val loss {hist.history['val_loss'][-1]:.4f}")
    for group, ids in [("val", val), ("test legit", test),
                       ("attack", [i for i, lab in enumerate(labels) if lab == "attack"]),
                       ("other", [i for i, lab in enumerate(labels) if lab == "other"])]:
        print(f"{group:>12}: median score {np.median(s[ids]):6.3f}, above threshold {np.mean(s[ids] > threshold):6.1%}")

    names = [f"{layer.name}.{v.name}" for layer in model.layers for v in layer.weights]
    w_json = {
        "tokenizer": name, "L": L, "vocab": vocab, "threshold": threshold, "epochs": EPOCHS,
        "params": int(sum(w.size for w in weights)), **extra,
        "names": names, "weights": [w.tolist() for w in weights],
        "history": {k: [round(float(v), 4) for v in hist.history[k]] for k in ("loss", "val_loss")},
    }
    return w_json, s


def main():
    lines = (ROOT / "data" / "auth.log").read_text(encoding="utf-8").splitlines()
    programs = {}
    for line in lines:
        m = PROGRAM.match(line)
        name = m.group(1).strip() if m else "?"
        programs[name] = programs.get(name, 0) + 1
    messages = [m.group(1) for line in lines if (m := SSHD.match(line))]
    labels = [label(msg) for msg in messages]
    train, val, test = split(labels)

    tf.config.experimental.enable_op_determinism()
    merges = bpe_train([messages[i] for i in train], BPE_MERGES)
    tokenizers = {"chars": (tok_chars, {}), "masked": (tok_masked, {}), "words": (tok_words, {}),
                  "bpe": (make_tok_bpe(merges), {"merges": merges})}

    (ROOT / "python").mkdir(exist_ok=True)
    all_weights, all_scores = {}, []
    for name, L in TOKENIZERS.items():
        tokenize, extra = tokenizers[name]
        w_json, s = train_one(name, L, tokenize, messages, labels, train, val, test, extra)
        (ROOT / "python" / f"weights_{name}.json").write_text(json.dumps(w_json, separators=(",", ":")) + "\n")
        all_weights[name] = w_json
        all_scores.append(s)

    (ROOT / "weights.js").write_text(
        "// Generated by tools/train_model.py. Don't edit by hand; change the script and rerun it.\n"
        "// One model per tokenizer, same content as python/weights_<tokenizer>.json: Keras weights rounded to 4 decimals, plus the loss curve.\n"
        "globalThis.AE_WEIGHTS = " + json.dumps(all_weights, separators=(",", ":")) + ";\n")

    split_of = {i: "train" for i in train} | {i: "val" for i in val} | {i: "test" for i in test}
    data = {
        "total": len(lines),
        "programs": dict(sorted(programs.items(), key=lambda kv: -kv[1])),
        "raw": lines[:6] + [line for line in lines if "Failed password" in line][:2],
        "tokenizers": list(TOKENIZERS),
        # One row per sshd message: [message, rule label, split, [score per tokenizer, with the rounded weights]]
        "lines": [[m, labels[i], split_of.get(i, "test" if labels[i] == "attack" else "none"),
                   [float(f"{s[i]:.7g}") for s in all_scores]]   # float32 precision: small scores matter
                  for i, m in enumerate(messages)],
    }
    (ROOT / "data.js").write_text(
        "// Generated by tools/train_model.py. Don't edit by hand.\n"
        "// The sshd messages of data/auth.log (Elastic examples, Apache-2.0) with their rule label, split and scores.\n"
        "globalThis.AE_DATA = " + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    old = ROOT / "python" / "weights.json"
    if old.exists():
        old.unlink()   # replaced by one file per tokenizer
    print("wrote weights.js, python/weights_*.json, data.js")


if __name__ == "__main__":
    main()
