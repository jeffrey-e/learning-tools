"""Trains the page's autoencoder with Keras and writes weights.js, python/weights.json and data.js.

Steps: keep the sshd lines of data/auth.log, strip the timestamp/host/pid prefix, label every
message with the rules below, split the legitimate messages into train/validation/test, train a
character-level Conv1D autoencoder on the training messages only, round the weights to 4 decimals,
and score every message with the rounded weights (so the page, the Python and data.js agree).

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
L = 128            # characters per line: longer messages are cut, shorter ones padded
EMBED, F1, F2, K = 8, 16, 4, 5
EPOCHS, BATCH, SEED = 150, 32, 0
SPLIT_SEED = 1


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


def build_model(V):
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
    """Mean of -log p(true character) over the real (non-padding) characters of each line."""
    P = model.predict(X, verbose=0, batch_size=256)
    p = np.take_along_axis(P, X[..., None], axis=2)[..., 0]
    nll = -np.log(np.maximum(p, 1e-7))
    mask = X > 0
    return (nll * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)


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

    vocab = ["<pad>", "<unk>"] + sorted(set("".join(messages[i] for i in train)))
    index = {c: i for i, c in enumerate(vocab)}

    def encode(msg):
        ids = [index.get(c, 1) for c in msg[:L]]
        return ids + [0] * (L - len(ids))

    X = np.array([encode(m) for m in messages], dtype="int32")
    mask = (X > 0).astype("float32")

    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    model = build_model(len(vocab))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    hist = model.fit(X[train], X[train], sample_weight=mask[train],
                     validation_data=(X[val], X[val], mask[val]),
                     epochs=EPOCHS, batch_size=BATCH, shuffle=True, verbose=0)

    weights = [np.round(w.astype(np.float64), 4) for w in model.get_weights()]
    model.set_weights([w.astype("float32") for w in weights])
    s = scores(model, X)
    threshold = round(float(np.percentile(s[val], 99)), 2)

    split_of = {i: "train" for i in train} | {i: "val" for i in val} | {i: "test" for i in test}
    for name, ids in [("train", train), ("val", val), ("test legit", test),
                      ("attack", [i for i, lab in enumerate(labels) if lab == "attack"]),
                      ("other", [i for i, lab in enumerate(labels) if lab == "other"])]:
        print(f"{name:>10}: {len(ids):5d} lines, median score {np.median(s[ids]):6.3f}, "
              f"above threshold {np.mean(s[ids] > threshold):6.1%}")
    print(f"threshold {threshold}, final loss {hist.history['loss'][-1]:.4f}, val loss {hist.history['val_loss'][-1]:.4f}")

    names = [f"{layer.name}.{v.name}" for layer in model.layers for v in layer.weights]
    w_json = {
        "L": L, "vocab": vocab, "threshold": threshold, "epochs": EPOCHS,
        "params": int(sum(w.size for w in weights)),
        "names": names, "weights": [w.tolist() for w in weights],
        "history": {k: [round(float(v), 4) for v in hist.history[k]] for k in ("loss", "val_loss")},
    }
    (ROOT / "python").mkdir(exist_ok=True)
    (ROOT / "python" / "weights.json").write_text(json.dumps(w_json, separators=(",", ":")) + "\n")
    (ROOT / "weights.js").write_text(
        "// Generated by tools/train_model.py. Don't edit by hand; change the script and rerun it.\n"
        "// Same content as python/weights.json: Keras weights rounded to 4 decimals, plus the loss curve.\n"
        "globalThis.AE_WEIGHTS = " + json.dumps(w_json, separators=(",", ":")) + ";\n")

    data = {
        "total": len(lines),
        "programs": dict(sorted(programs.items(), key=lambda kv: -kv[1])),
        "raw": lines[:6] + [lines[i] for i in range(len(lines)) if "Failed password" in lines[i]][:2],
        # One row per sshd message: [message, rule label, split, score with the rounded weights]
        "lines": [[m, labels[i], split_of.get(i, "test" if labels[i] == "attack" else "none"), round(float(s[i]), 4)]
                  for i, m in enumerate(messages)],
    }
    (ROOT / "data.js").write_text(
        "// Generated by tools/train_model.py. Don't edit by hand.\n"
        "// The sshd messages of data/auth.log (Elastic examples, Apache-2.0) with their rule label, split and score.\n"
        "globalThis.AE_DATA = " + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    print("wrote weights.js, python/weights.json, data.js")


if __name__ == "__main__":
    main()
