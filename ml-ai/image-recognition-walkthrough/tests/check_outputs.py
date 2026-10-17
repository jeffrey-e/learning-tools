"""Runs every Keras cell the page shows (dumped by dump_states.js) and checks that the printed output
is what the page claims: exactly, or with numbers at most one off in their last printed digit
(different processor types round differently; such cases are listed). Also checks that python/walkthrough.py is up to date and runs, runs
each training cell for one epoch, and writes tests/tfjs_ref.json (decoded pixels of a few photos and
the Keras predictions) for check_tfjs.js.

Usage:  python3 tests/check_outputs.py [--quick]   (after: node tests/dump_states.js)
        --quick skips the training cells.
"""
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import re
import subprocess
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def last_digit_only(page, got):
    """True when the two outputs have the same text and every decimal number differs by at most one in its
    last printed digit (whole numbers, like counts, must match exactly). TensorFlow's float maths differs slightly between processor types (x86 with oneDNN vs
    ARM), so a value right on a rounding boundary can print as 0.687 on one and 0.688 on the other."""
    if NUMBER.sub("#", page) != NUMBER.sub("#", got):
        return False
    for a, b in zip(NUMBER.findall(page), NUMBER.findall(got)):
        decimals = len(a.split(".")[1]) if "." in a else 0
        if decimals == 0 and a != b:   # whole numbers (counts, shapes) must match exactly
            return False
        if ("." in b and len(b.split(".")[1]) != decimals) or abs(float(a) - float(b)) > 1.0001 * 10 ** -decimals:
            return False
    return True


def prefetch():
    """Downloads (and checks the pinned hashes of) the pretrained weights and the ImageNet class list before
    any cell runs, so no "Downloading data from ..." message ends up in a cell's printed output."""
    import keras
    import numpy as np
    spec = importlib.util.spec_from_file_location("pretrained", ROOT / "tools" / "pretrained.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.verify(keras)
    keras.applications.mobilenet_v2.decode_predictions(np.zeros((1, 1000)), top=1)


def run_cell(code, ns):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, ns)
    return buf.getvalue().rstrip("\n")


def main():
    quick = "--quick" in sys.argv
    os.chdir(ROOT)   # the cells read images/, data/ and python/
    data = json.loads((ROOT / "tests" / "states.json").read_text())
    prefetch()
    failures, outputs, matched, trained, ref, near = [], 0, 0, set(), [], []
    for state in data["states"]:
        ns = {"__name__": "__main__"}
        for cell in state["cells"]:
            if cell.get("train"):
                if not quick and cell["id"] not in trained:
                    code = cell["code"].replace("epochs=", "epochs=1 or ")
                    run_cell(code, ns)
                    trained.add(cell["id"])
                continue
            got = run_cell(cell["code"], ns)
            outputs += 1
            if got == cell["out"]:
                matched += 1
            elif last_digit_only(cell["out"], got):
                matched += 1
                near.append(f"{state['name']} / {cell['id']}")
            else:
                failures.append(f"{state['name']} / {cell['id']}:\n--- page\n{cell['out']}\n--- keras\n{got}")
        ref.append({"photo": ns["photo"], "pixels": ns["x"][0].astype(int).flatten().tolist(),
                    "pa": float(ns["scratch"].predict(ns["x"], verbose=0)[0, 1]),
                    "pb": float(ns["p"][1])})
    (ROOT / "tests" / "tfjs_ref.json").write_text(json.dumps(ref))

    script = ROOT / "python" / "walkthrough.py"
    if script.read_text() != data["script"]:
        failures.append("python/walkthrough.py is out of date: run npm run generate")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        failures.append(f"python/walkthrough.py failed:\n{result.stderr[-2000:]}")

    print(f"page outputs: {matched}/{outputs} match real Keras" + (f" ({len(near)} differ only in the last digit: {', '.join(near)})" if near else ""))
    if failures:
        print("\n\n".join(failures))
        sys.exit(1)
    print("python/walkthrough.py: ran")
    print("training cells: " + ("skipped (--quick)" if quick else f"ran {len(trained)} for 1 epoch"))
    print("all checks passed")


if __name__ == "__main__":
    main()
