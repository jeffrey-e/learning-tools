"""Converts the two trained models to TensorFlow.js for the page: model/scratch/ and model/transfer/.

Runs in its own environment (see requirements-export.txt): the TF.js converter needs TensorFlow 2.15
and tf_keras, while training uses current Keras. Both models are rebuilt here with extra outputs the
page shows (early and late feature maps), the trained weights are loaded from
python/export_weights.npz, the predictions are checked against data.js, and the models are written
in the TF.js layers format (model B in float16, half the download, about 0.001 off in probability).

Only the converter's Keras module is imported: the full tensorflowjs package pulls in optional
dependencies (JAX, TF Decision Forests) that are not needed here and don't install everywhere.

Usage:  <export env>/bin/python tools/export_tfjs.py
"""
import importlib
import importlib.util
import json
import os
import pathlib
import site
import sys
import tempfile
import types

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import tf_keras as keras  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
_pkg = os.path.join(site.getsitepackages()[0], "tensorflowjs")
for _name, _path in [("tensorflowjs", _pkg), ("tensorflowjs.converters", os.path.join(_pkg, "converters"))]:
    _mod = types.ModuleType(_name)
    _mod.__path__ = [_path]
    sys.modules[_name] = _mod
h5_to_tfjs = importlib.import_module("tensorflowjs.converters.keras_h5_conversion")
L = keras.layers


def scratch_model(cfg):
    """Model A, as in tools/experiments.py, returning [first feature maps, last feature maps, probabilities]."""
    inp = keras.Input((224, 224, 3))
    x = L.Rescaling(1 / 255, name="rescale")(inp)
    if cfg["size"] != 224:
        x = L.Resizing(cfg["size"], cfg["size"], name="resize")(x)
    maps = []
    for i, f in enumerate(cfg["widths"]):
        x = L.Conv2D(f, 3, padding="same", use_bias=not cfg["bn"], name=f"conv{i + 1}")(x)
        if cfg["bn"]:
            x = L.BatchNormalization(name=f"bn{i + 1}")(x)
        x = L.ReLU(name=f"relu{i + 1}")(x)
        maps.append(x)
        if i < len(cfg["widths"]) - 1:
            x = L.MaxPooling2D(name=f"pool{i + 1}")(x)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(0.3, name="dropout")(x)
    out = L.Dense(2, activation="softmax", name="classify")(x)
    return keras.Model(inp, [maps[0], maps[-1], out], name="scratch")


def transfer_model(cfg):
    """Model B: MobileNetV2 (ImageNet weights) + the trained head, returning
    [first conv layer, a middle layer, last feature maps, probabilities]."""
    inp = keras.Input((224, 224, 3))
    x = L.Rescaling(1 / 127.5, offset=-1, name="preprocess")(inp)   # = mobilenet_v2.preprocess_input
    base = keras.applications.MobileNetV2(input_tensor=x, include_top=False, weights="imagenet")
    feat = base.get_layer("out_relu").output
    g = L.GlobalAveragePooling2D(name="gap")(feat)
    g = L.Dropout(cfg["dropout"], name="dropout")(g)
    out = L.Dense(2, activation="softmax", name="classify")(g)
    return keras.Model(inp, [base.get_layer("Conv1_relu").output, base.get_layer("block_6_expand_relu").output, feat, out], name="transfer")


def verify_pretrained():
    """Checks the pinned SHA-256 of the MobileNetV2 weights before tf_keras uses them (tools/pretrained.py)."""
    spec = importlib.util.spec_from_file_location("pretrained", ROOT / "tools" / "pretrained.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.verify(keras)


def main():
    verify_pretrained()
    cfg = json.loads((ROOT / "python" / "export_config.json").read_text())
    w = np.load(ROOT / "python" / "export_weights.npz")
    data = json.loads((ROOT / "data.js").read_text().split("globalThis.IMG_DATA = ", 1)[1].rstrip().rstrip(";"))

    a = scratch_model(cfg["scratch"])
    a.set_weights([w[f"scratch_{i}"] for i in range(sum(k.startswith("scratch_") for k in w.files))])
    b = transfer_model(cfg["transfer"])
    head = [w[f"head_{i}"] for i in range(sum(k.startswith("head_") for k in w.files))]
    b.get_layer("classify").set_weights(head)

    # The rebuilt models must give the predictions train.py stored (same weights, same maths)
    test = [p for p in data["photos"] if p["split"] == "test"][:12]
    X = np.stack([np.asarray(Image.open(ROOT / p["file"]).convert("RGB"), dtype="float32") for p in test])
    da = np.abs(a.predict(X, verbose=0)[-1][:, 1] - [p["pa"] for p in test]).max()
    db = np.abs(b.predict(X, verbose=0)[-1][:, 1] - [p["pb"] for p in test]).max()
    print(f"rebuilt models vs data.js: model A {da:.1e}, model B {db:.1e}")
    if max(da, db) > 1e-4:
        sys.exit("the rebuilt models don't match the trained ones")

    for name, model, dtype in (("scratch", a, None), ("transfer", b, "float16")):
        out = ROOT / "model" / name
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name + ".h5")
            model.save(path)
            h5_to_tfjs.save_keras_model(keras.models.load_model(path, compile=False), str(out),
                                        quantization_dtype_map={dtype: "*"} if dtype else None)
        size = sum(f.stat().st_size for f in out.iterdir())
        print(f"wrote model/{name}/ ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
