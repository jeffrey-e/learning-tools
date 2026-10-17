"""Runs every Keras cell the page shows (dumped by dump_states.js) and checks that the printed output
is exactly what the page claims. Also checks that python/walkthrough.py is up to date and runs, runs
each training cell for one epoch, and writes tests/tfjs_ref.json (decoded pixels of a few photos and
the Keras predictions) for check_tfjs.js.

Usage:  python3 tests/check_outputs.py [--quick]   (after: node tests/dump_states.js)
        --quick skips the training cells.
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


def run_cell(code, ns):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, ns)
    return buf.getvalue().rstrip("\n")


def main():
    quick = "--quick" in sys.argv
    os.chdir(ROOT)   # the cells read images/, data/ and python/
    data = json.loads((ROOT / "tests" / "states.json").read_text())
    failures, outputs, matched, trained, ref = [], 0, 0, set(), []
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

    print(f"page outputs: {matched}/{outputs} match real Keras")
    if failures:
        print("\n\n".join(failures))
        sys.exit(1)
    print("python/walkthrough.py: ran")
    print("training cells: " + ("skipped (--quick)" if quick else f"ran {len(trained)} for 1 epoch"))
    print("all checks passed")


if __name__ == "__main__":
    main()
