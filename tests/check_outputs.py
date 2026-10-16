"""Runs every Python cell the page shows (dumped by dump_states.js) in real NumPy and TensorFlow
and checks that the printed output is exactly what the page claims. Also runs each copyable full
script, every script in python/, and a short version of each training script.

Usage:  python tests/check_outputs.py          (after: node tests/dump_states.js)
        python tests/check_outputs.py --quick  (skip TensorFlow training, the slow part)
"""
import contextlib
import io
import json
import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: F401  (imported so failures to install show up early)
import tensorflow as tf  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parent.parent
quick = "--quick" in sys.argv
data = json.loads((ROOT / "tests" / "states.json").read_text())


def run(code):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {"__name__": "__main__"})
    return buf.getvalue().rstrip("\n")


failures, checked = [], 0
for st in data["states"]:
    ns = {}
    for cell in st["cells"]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(cell["code"], ns)
        got = buf.getvalue().rstrip("\n")
        if cell["out"] is not None:
            checked += 1
            if got != cell["out"]:
                failures.append(f"[{st['mode']}] {st['label']}\n--- code ---\n{cell['code']}\n"
                                f"--- page shows ---\n{cell['out']}\n--- Python prints ---\n{got}")
    run(st["full"])
print(f"page outputs: {checked - len(failures)}/{checked} match real NumPy/TensorFlow; "
      f"{len(data['states'])} full scripts ran")

for script in sorted((ROOT / "python").glob("*.py")):
    if quick and script.name == "train_tensorflow.py":
        continue
    code = script.read_text().replace("epochs=100", "epochs=5").replace("epochs, batch = 0.3, 100, 8", "epochs, batch = 0.3, 5, 8")
    run(code)
    print(f"python/{script.name}: ran")

for t in data["training"]:
    if quick and t["mode"] == "tf":
        continue
    code = t["code"].replace("epochs=100", "epochs=3").replace(", 100, 8", ", 3, 8")
    out = run(code)
    if "accuracy" not in out:
        failures.append(f"training script {t['mode']}/{t['opt']} printed no accuracy:\n{out}")
    print(f"training script {t['mode']}/{t['opt']}: ran")

if failures:
    print("\n\n".join(failures))
    sys.exit(1)
print("all checks passed")
