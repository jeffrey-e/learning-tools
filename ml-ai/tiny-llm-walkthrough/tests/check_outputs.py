"""Runs every NumPy cell the page shows (dumped by dump_states.js) and checks that the printed
output is exactly what the page claims, and that the arrays match the page's JavaScript.
Also checks that python/llm_numpy.py is up to date and runs.

Usage:  python tests/check_outputs.py   (after: node tests/dump_states.js)
"""
import contextlib
import io
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOLERANCE = 1e-9


def run_cell(code, namespace):
    buf = io.StringIO()
    # errstate: an all-zero embedding row has no direction, so its cosine similarity is nan (on the page too)
    with contextlib.redirect_stdout(buf), np.errstate(invalid="ignore"):
        exec(code, namespace)
    return buf.getvalue().rstrip("\n")


def main():
    data = json.loads((ROOT / "tests" / "states.json").read_text())
    failures, outputs, matched = [], 0, 0

    for state in data["states"]:
        ns = {"__name__": "__main__"}
        for cell in state["cells"]:
            got = run_cell(cell["code"], ns)
            outputs += 1
            if got == cell["out"]:
                matched += 1
            else:
                failures.append(f"{state['name']} / {cell['id']}:\n--- page\n{cell['out']}\n--- numpy\n{got}")
        for name, expected in state["arrays"].items():
            actual = ns[name]
            if not np.allclose(actual, np.array(expected), atol=TOLERANCE, rtol=0):
                failures.append(f"{state['name']} / array {name} differs from the page")

    script_path = ROOT / "python" / "llm_numpy.py"
    if script_path.read_text() != data["script"]:
        failures.append("python/llm_numpy.py is out of date: run npm run generate")
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        failures.append(f"python/llm_numpy.py failed:\n{result.stderr}")

    print(f"page outputs: {matched}/{outputs} match real NumPy")
    if failures:
        print("\n\n".join(failures))
        sys.exit(1)
    print("python/llm_numpy.py: ran")
    print("all checks passed")


if __name__ == "__main__":
    main()
