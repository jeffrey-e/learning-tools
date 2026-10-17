"""Runs every Keras cell the page shows (dumped by dump_states.js) and checks that the printed
output is exactly what the page claims, and that the arrays match the page's JavaScript.
Also checks the split and the precomputed scores in data.js, that python/autoencoder_tf.py is up
to date and runs, and runs the training cell for one epoch.

Usage:  python tests/check_outputs.py [--quick]   (after: node tests/dump_states.js)
        --quick skips the training cell.
"""
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOLERANCE = 1e-4   # TensorFlow computes in float32, the page in float64


def run_cell(code, namespace):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, namespace)
    return buf.getvalue().rstrip("\n")


def main():
    quick = "--quick" in sys.argv
    os.chdir(ROOT)   # the cells read data/auth.log and python/weights.json
    data = json.loads((ROOT / "tests" / "states.json").read_text())
    failures, outputs, matched, trained = [], 0, 0, set()

    for state in data["states"]:
        ns = {"__name__": "__main__"}
        for cell in state["cells"]:
            if cell.get("train"):
                if quick or state["name"].split(" | ")[0] in trained:
                    continue
                # The real run takes 150 epochs; one is enough to show the code works.
                run_cell(cell["code"].replace("epochs=", "epochs=1 or "), ns)
                trained.add(state["name"].split(" | ")[0])
                continue
            got = run_cell(cell["code"], ns)
            outputs += 1
            if got == cell["out"]:
                matched += 1
            else:
                failures.append(f"{state['name']} / {cell['id']}:\n--- page\n{cell['out']}\n--- tensorflow\n{got}")
        for name, expected in state["arrays"].items():
            actual = ns["outs"][name][: len(expected)]
            if not np.allclose(actual, np.array(expected), atol=TOLERANCE, rtol=0):
                failures.append(f"{state['name']} / array {name} differs from the page")
        if state.get("first") is not None:
            for split, ids in data["splits"].items():
                if sorted(ns[f"{split}_ids"]) != ids:
                    failures.append(f"the {split} split in the page's code differs from data.js")
            if not np.allclose(ns["scores"], data["scores"][state["first"]], atol=TOLERANCE * 10, rtol=0):
                failures.append(f"{state['name']}: the page's scores differ from data.js")

    script_path = ROOT / "python" / "autoencoder_tf.py"
    if script_path.read_text() != data["script"]:
        failures.append("python/autoencoder_tf.py is out of date: run npm run generate")
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        failures.append(f"python/autoencoder_tf.py failed:\n{result.stderr}")

    print(f"page outputs: {matched}/{outputs} match real TensorFlow")
    if failures:
        print("\n\n".join(failures))
        sys.exit(1)
    print("python/autoencoder_tf.py: ran")
    print("training cell: " + ("skipped (--quick)" if quick else f"ran for 1 epoch with {len(trained)} tokenizers"))
    print("all checks passed")


if __name__ == "__main__":
    main()
