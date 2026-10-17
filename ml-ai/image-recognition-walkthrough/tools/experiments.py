"""The tuning experiments behind the two models, rerunnable. Writes data/experiments.json, which the
page shows as a table and tools/train.py reads to build the chosen configurations.

Every variant is judged on the validation set only. The test set is used once, by tools/train.py,
for the chosen models. "Balanced" accuracy is the mean of the husky and the malamute recall: a model
that always answers "husky" scores about 75% plain accuracy here, but 50% balanced.

Usage:  python3 -P tools/experiments.py      (about half an hour on a CPU; seeded and deterministic)
"""
import json
import os
import pathlib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED = 0


def verify_pretrained():
    """Checks the pinned SHA-256 of the MobileNetV2 weights before Keras uses them (tools/pretrained.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pretrained", ROOT / "tools" / "pretrained.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.verify(keras)

BASE_A = {"bn": False, "size": 224, "widths": [16, 32, 64, 128], "lr": 1e-3, "epochs": 60, "augment": True, "class_weight": True}
A_RUNS = [
    ("A1", "First try", {}, "A plain small CNN with augmentation and class weights. The loss barely moves from 0.69, the loss of guessing."),
    ("A2", "+ batch normalisation", {"bn": True}, "Normalising each layer's output usually helps a small network start learning."),
    ("A3", "+ 128px input, 100 epochs", {"bn": True, "size": 128, "epochs": 100}, "Fewer pixels: faster, and less detail to overfit on."),
    ("A4", "+ 4× wider", {"bn": True, "size": 128, "epochs": 100, "widths": [32, 64, 128, 256]}, "More filters per layer: is the network too small?"),
    ("A5", "Lower learning rate, no augmentation", {"bn": True, "size": 128, "epochs": 150, "lr": 3e-4, "augment": False},
     "A test, not a candidate: can it at least memorise the training photos?"),
    ("A6", "Lower learning rate, with augmentation", {"bn": True, "size": 128, "epochs": 150, "lr": 3e-4},
     "The same, with flips, rotations and zooms so it can't simply memorise."),
    ("A7", "64px input", {"bn": True, "size": 64, "epochs": 150, "lr": 3e-4}, "Even less detail: maybe coat colour and shape are enough."),
]
BASE_B = {"lr": 1e-3, "epochs": 80, "dropout": 0.2, "class_weight": True}
B_RUNS = [
    ("B1", "Default head", {}, "Dropout 0.2, Adam with learning rate 0.001, 80 epochs, class weights."),
    ("B2", "No dropout", {"dropout": 0.0}, "Does the head need regularising at all?"),
    ("B3", "More dropout", {"dropout": 0.5}, "Stronger regularisation."),
    ("B4", "10× learning rate", {"lr": 1e-2}, "Faster training: does it still settle?"),
    ("B5", "No class weights", {"class_weight": False}, "Without weighing up the rarer malamutes."),
]


def build_scratch(cfg):
    """Model A for a configuration. Same layers as tools/train.py and tools/export_tfjs.py."""
    L = [keras.Input((224, 224, 3)), layers.Rescaling(1 / 255, name="rescale")]
    if cfg["size"] != 224:
        L.append(layers.Resizing(cfg["size"], cfg["size"], name="resize"))
    for i, f in enumerate(cfg["widths"]):
        L.append(layers.Conv2D(f, 3, padding="same", use_bias=not cfg["bn"], name=f"conv{i + 1}"))
        if cfg["bn"]:
            L.append(layers.BatchNormalization(name=f"bn{i + 1}"))
        L.append(layers.ReLU(name=f"relu{i + 1}"))
        if i < len(cfg["widths"]) - 1:
            L.append(layers.MaxPooling2D(name=f"pool{i + 1}"))
    L += [layers.GlobalAveragePooling2D(name="gap"), layers.Dropout(0.3, name="dropout"), layers.Dense(2, activation="softmax", name="classify")]
    return keras.Sequential(L, name="scratch")


def build_head(cfg):
    return keras.Sequential([keras.Input((1280,)), layers.Dropout(cfg["dropout"], name="dropout"),
                             layers.Dense(2, activation="softmax", name="classify")], name="head")


def load():
    photos = json.loads((ROOT / "data" / "photos.json").read_text())
    X = np.stack([np.asarray(Image.open(ROOT / p["file"]).convert("RGB"), dtype="float32") for p in photos])
    y = np.array([p["label"] == "malamute" for p in photos]).astype(int)
    split = np.array([p["split"] for p in photos])
    return photos, X, y, split


def class_weights(y):
    return {k: len(y) / (2 * (y == k).sum()) for k in range(2)}


def augmenter():
    return keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08), layers.RandomZoom(0.15)], name="augment")


def train_scratch(cfg, X, y, tr, va):
    keras.utils.set_random_seed(SEED)
    aug = augmenter()
    ds = tf.data.Dataset.from_tensor_slices((X[tr], y[tr])).shuffle(512, seed=SEED).batch(16)
    if cfg["augment"]:
        ds = ds.map(lambda a, b: (aug(a, training=True), b))
    model = build_scratch(cfg)
    model.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(ds, validation_data=(X[va], y[va]), epochs=cfg["epochs"],
                     class_weight=class_weights(y[tr]) if cfg["class_weight"] else None, verbose=0)
    return model, hist


def train_head(cfg, F, y, tr, va):
    keras.utils.set_random_seed(SEED)
    model = build_head(cfg)
    model.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(F[tr], y[tr], validation_data=(F[va], y[va]), epochs=cfg["epochs"], batch_size=16, shuffle=True,
                     class_weight=class_weights(y[tr]) if cfg["class_weight"] else None, verbose=0)
    return model, hist


def features(X):
    verify_pretrained()
    base = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling="avg", weights="imagenet")
    return base, base.predict(keras.applications.mobilenet_v2.preprocess_input(X.copy()), verbose=0, batch_size=32)


def scores(p, y, mask):
    pred = p[mask].argmax(1)
    return {"accuracy": round(float((pred == y[mask]).mean()), 3),
            "balanced": round(float(np.mean([(pred[y[mask] == k] == k).mean() for k in range(2)])), 3)}


def main():
    tf.config.experimental.enable_op_determinism()
    _, X, y, split = load()
    tr, va = split == "train", split == "val"
    out = {"scratch": [], "transfer": []}
    for rid, name, change, why in A_RUNS:
        cfg = {**BASE_A, **change}
        model, hist = train_scratch(cfg, X, y, tr, va)
        p = model.predict(X, verbose=0, batch_size=32)
        row = {"id": rid, "name": name, "why": why, "config": cfg, "params": int(model.count_params()),
               "final_loss": round(hist.history["loss"][-1], 3),
               "train": scores(p, y, tr), "val": scores(p, y, va)}
        out["scratch"].append(row)
        print(rid, name, row["train"], row["val"], flush=True)
    _, F = features(X)
    for rid, name, change, why in B_RUNS:
        cfg = {**BASE_B, **change}
        model, hist = train_head(cfg, F, y, tr, va)
        p = model.predict(F, verbose=0)
        row = {"id": rid, "name": name, "why": why, "config": cfg, "params": int(model.count_params()),
               "final_loss": round(hist.history["loss"][-1], 3),
               "train": scores(p, y, tr), "val": scores(p, y, va)}
        out["transfer"].append(row)
        print(rid, name, row["train"], row["val"], flush=True)
    # The choice: the best balanced validation accuracy; on a tie the earlier (simpler) run. A5 is a test, not a candidate.
    pick = lambda rows: max((r for r in rows if r["id"] != "A5"), key=lambda r: (r["val"]["balanced"], -rows.index(r)))["id"]
    out["chosen"] = {"scratch": pick(out["scratch"]), "transfer": pick(out["transfer"])}
    (ROOT / "data" / "experiments.json").write_text(json.dumps(out, indent=1) + "\n")
    print("chosen:", out["chosen"])


if __name__ == "__main__":
    main()
