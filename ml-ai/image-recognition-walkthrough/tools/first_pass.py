"""First pass over the candidates, before the human review in tools/review.html.

Drops photos where a pretrained ImageNet MobileNetV2 sees no dog at all (no dog class in its top 5:
parachutes, helicopters, landscapes). It does NOT decide the breed: the label stays the breed the
photo was found under, so the labels don't come from the kind of model the walkthrough tests.
Labels set in the review (by: "user") are never overwritten.

Usage:  python3 -P tools/first_pass.py
"""
import importlib.util
import json
import os
import pathlib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOGS = range(151, 269)   # ImageNet classes 151-268 are dog breeds


def main():
    cands = json.loads((ROOT / "data" / "candidates.json").read_text())
    path = ROOT / "data" / "labels.json"
    labels = json.loads(path.read_text()) if path.exists() else {}
    spec = importlib.util.spec_from_file_location("pretrained", ROOT / "tools" / "pretrained.py")
    pretrained = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pretrained)
    pretrained.verify(keras)   # pinned SHA-256 of the ImageNet weights
    model = keras.applications.MobileNetV2(weights="imagenet")
    decode = keras.applications.mobilenet_v2.decode_predictions

    def load(c):
        im = Image.open(ROOT / "data" / "candidates" / (c["id"] + ".jpg")).convert("RGB").resize((224, 224))
        return keras.applications.mobilenet_v2.preprocess_input(np.asarray(im, dtype="float32"))

    for i in range(0, len(cands), 64):
        batch = cands[i:i + 64]
        P = model.predict(np.stack([load(c) for c in batch]), verbose=0)
        for c, p, top in zip(batch, P, decode(P, top=5)):
            if labels.get(c["id"], {}).get("by") == "user":
                continue
            dog = any(k in DOGS for k in np.argsort(-p)[:5])
            search_only = all(s.startswith("search") for s in c["sources"])
            if not dog:
                labels[c["id"]] = {"label": "drop", "by": "first-pass", "note": "no dog seen (top: " + top[0][1].replace("_", " ") + ")"}
            elif c["guess"] == "both":
                labels[c["id"]] = {"label": "drop", "by": "first-pass", "note": "found under both breeds"}
            else:
                labels[c["id"]] = {"label": c["guess"], "by": "first-pass", "note": "from search only: check" if search_only else ""}
    path.write_text(json.dumps(labels, indent=1, sort_keys=True) + "\n")
    n = {}
    for v in labels.values():
        n[v["label"]] = n.get(v["label"], 0) + 1
    print(n)


if __name__ == "__main__":
    main()
