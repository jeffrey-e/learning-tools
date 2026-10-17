"""Trains the two final models with the configurations chosen in tools/experiments.py, and writes
everything the page and the Python cells use.

Model A, from scratch: a small CNN trained only on our training photos (the chosen experiment).
Model B, transfer learning: MobileNetV2 pretrained on ImageNet, frozen. Its 1280 features per photo
are computed once; only a new head (1280 -> 2) is trained.
Both end in global average pooling + one Dense layer, so the class activation map (what the model
looks at) is exactly a weighted sum of the last feature maps.

The test set is used here, once, after the configurations were chosen on the validation set.

Writes: python/scratch.weights.h5 and python/transfer_head.weights.h5 (for the Python cells),
python/export_weights.npz and python/export_config.json (for tools/export_tfjs.py), and data.js.

Usage:  python3 -P tools/train.py        (seeded and deterministic on CPU)
"""
import importlib.util
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
# The model builders and training loops are shared with the experiments, so the chosen run is exactly what's trained here.
_spec = importlib.util.spec_from_file_location("experiments", ROOT / "tools" / "experiments.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

CLASSES = ["husky", "malamute"]


def block_name(n):
    """Groups Keras layers into the blocks the page draws (same rules as the page's Python cell)."""
    m = re.match(r"block_(\d+)_", n)
    if m:
        return "block " + m.group(1)
    if n.startswith("expanded_conv"):
        return "block 0"
    if n in ("Conv1", "bn_Conv1", "Conv1_relu"):
        return "stem"
    if n in ("Conv_1", "Conv_1_bn", "out_relu"):
        return "last conv"
    m = re.match(r"(conv|bn|relu|pool)(\d+)$", n)
    if m:
        return "conv " + m.group(2)
    if n == "gap" or n.startswith("global_average_pooling"):
        return "average pooling"
    return n


def blocks(layers, part):
    """One row per block: [name, output shape, weights, frozen/trained, the Keras layers in it]."""
    rows = []
    for layer in layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        name, shape = block_name(layer.name), list(layer.output.shape)[1:]
        if rows and rows[-1][0] == name:
            rows[-1][1], rows[-1][2] = shape, rows[-1][2] + layer.count_params()
            rows[-1][4].append(layer.name)
        else:
            rows.append([name, shape, layer.count_params(), part, [layer.name]])
    return rows


def main():
    tf.config.experimental.enable_op_determinism()
    exp = json.loads((ROOT / "data" / "experiments.json").read_text())
    cfg_a = next(r for r in exp["scratch"] if r["id"] == exp["chosen"]["scratch"])["config"]
    cfg_b = next(r for r in exp["transfer"] if r["id"] == exp["chosen"]["transfer"])["config"]
    photos, X, y, split = E.load()
    tr, va, te = split == "train", split == "val", split == "test"

    scratch, hist_a = E.train_scratch(cfg_a, X, y, tr, va)
    base, F = E.features(X)
    head, hist_b = E.train_head(cfg_b, F, y, tr, va)

    # Predictions one photo at a time, exactly as the page's code cells run them (a batch of 32 can differ
    # in the last float digit, enough to flip a rounding)
    pa = scratch.predict(X, verbose=0, batch_size=1)
    F1 = base.predict(keras.applications.mobilenet_v2.preprocess_input(X.copy()), verbose=0, batch_size=1)
    pb = head.predict(F1, verbose=0, batch_size=1)
    prep = keras.applications.mobilenet_v2.preprocess_input
    # Class activation maps of model B for the test photos: last feature maps (7x7x1280) weighted by the head's Dense weights
    fmap = keras.Model(base.input, base.get_layer("out_relu").output)
    W = head.get_layer("classify").get_weights()[0]                       # (1280, 2)
    A = fmap.predict(prep(X[te].copy()), verbose=0, batch_size=1)         # (n, 7, 7, 1280)
    cams = A @ W                                                          # (n, 7, 7, 2)
    # What the original 1000-class ImageNet model says (it knows both breeds, and "Eskimo dog")
    full = keras.applications.MobileNetV2(weights="imagenet")
    top = keras.applications.mobilenet_v2.decode_predictions(full.predict(prep(X.copy()), verbose=0, batch_size=1), top=3)

    def report(p, name):
        out = {}
        for s, m in (("val", va), ("test", te)):
            pred = p[m].argmax(1)
            cm = [[int(((y[m] == i) & (pred == j)).sum()) for j in range(2)] for i in range(2)]
            out[s] = {**E.scores(p, y, m), "confusion": cm}
        print(f"{name}: test {out['test']}")
        return out

    metrics = {"scratch": report(pa, "A from scratch (" + exp["chosen"]["scratch"] + ")"),
               "transfer": report(pb, "B transfer learning (" + exp["chosen"]["transfer"] + ")")}

    (ROOT / "python").mkdir(exist_ok=True)
    scratch.save_weights(ROOT / "python" / "scratch.weights.h5")
    head.save_weights(ROOT / "python" / "transfer_head.weights.h5")
    np.savez(ROOT / "python" / "export_weights.npz",
             **{f"scratch_{i}": w for i, w in enumerate(scratch.get_weights())},
             **{f"head_{i}": w for i, w in enumerate(head.get_weights())})
    (ROOT / "python" / "export_config.json").write_text(json.dumps({"scratch": cfg_a, "transfer": cfg_b}, indent=1) + "\n")

    hist = lambda h: {k: [round(float(v), 4) for v in h.history[k]] for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
    g7 = lambda v: float(f"{v:.7g}")   # enough digits that printing to 3 decimals never double-rounds
    test_index = {i: n for n, i in enumerate(np.flatnonzero(te))}
    data = {
        "classes": CLASSES,
        "counts": {s: {c: int(((split == s) & (y == i)).sum()) for i, c in enumerate(CLASSES)} for s in ("train", "val", "test")},
        "chosen": exp["chosen"], "config": {"scratch": cfg_a, "transfer": cfg_b},
        "params": {"scratch": int(scratch.count_params()), "base": int(base.count_params()), "head": int(head.count_params())},
        "history": {"scratch": hist(hist_a), "transfer": hist(hist_b)},
        # The structure of both models, block by block, for the page's diagram (step 5)
        "arch": {"scratch": blocks(scratch.layers, "trained"),
                 "transfer": blocks(base.layers, "frozen") + blocks(head.layers, "trained")},
        "metrics": metrics,
        "experiments": {k: exp[k] for k in ("scratch", "transfer")},
        # One row per photo: credits, split, the Keras probability of "malamute" for both models, ImageNet's top 3,
        # and for test photos model B's class activation map for the predicted class (7x7)
        "photos": [{**{k: p[k] for k in ("id", "label", "split", "file", "title", "author", "licence", "licence_url", "page")},
                    "px": [int(v) for v in X[i, 112, 112]],   # the centre pixel as PIL decodes it (browsers may differ by 1)
                    "pa": g7(pa[i, 1]), "pb": g7(pb[i, 1]),
                    "imagenet": [[t[1].replace("_", " "), g7(t[2])] for t in top[i]],
                    **({"cam": [[g7(v) for v in row] for row in cams[test_index[i], :, :, int(pb[i].argmax())]]} if te[i] else {})}
                   for i, p in enumerate(photos)],
    }
    (ROOT / "data.js").write_text(
        "// Generated by tools/train.py. Don't edit by hand.\n"
        "// Photos from Wikimedia Commons (see the credits on the page), with the Keras predictions of both models.\n"
        "globalThis.IMG_DATA = " + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    print("wrote data.js, python/*.weights.h5, python/export_*")


if __name__ == "__main__":
    main()
