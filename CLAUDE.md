# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A teaching page that walks one 6×6 image through a tiny CNN (2 kernels of 3×3, ReLU, 2×2 max pool, flatten, dense 8→4, softmax), with editable weights, origin labels for every value, live NumPy/TensorFlow code with verified outputs, and a train mode. Hosted on GitHub Pages straight from the `main` branch root. There is no build step.

## Layout

- `index.html`: the entire page. CSS, HTML and JavaScript inline; the only external resources are Google Fonts and TensorFlow.js 4.22.0 from jsDelivr (pinned).
- `python/`, `notebooks/`: **generated** from `index.html`. Don't edit by hand; change the page, then run `npm run generate`.
- `tests/dump_states.js`: drives the page headlessly (jsdom, with TensorFlow.js from `node_modules`) through 18 states and dumps every code cell and printed output to `tests/states.json`.
- `tests/check_outputs.py`: runs that code in real NumPy and TensorFlow and fails on any difference. Also runs `python/` and the training scripts with a few epochs.
- `tools/generate_python.js`, `tools/build_notebook.py`: the generators.

## Commands

```bash
npm ci && pip install -r requirements.txt        # once (plus: pip install nbformat nbclient ipykernel for the notebook)
npm test                                         # full verification (about a minute, mostly TensorFlow)
npm run test:quick                               # skips TensorFlow training
npm run generate                                 # regenerate python/ and the notebook after changing index.html
```

Run `npm test` after every change to `index.html`. A change is not done until it passes.

## How index.html's script is organised

- **Engines.** `runNP` (plain JavaScript mirroring NumPy) and `runTF` (TensorFlow.js ops) compute conv, activation and pooling; `dense()` does logits and softmax. `useTF()` picks one. If TensorFlow.js fails to load, the page falls back to plain JavaScript and says so (`setEngineNote`).
- **NumPy-style printing.** `npStr`, `decStr`, `pyRound4` reproduce NumPy's array and scalar printing, including float32 shortest repr, capped at 4 decimals to match `np.set_printoptions(precision=4, suppress=True)` in the generated code. `pyLit` writes Python literals.
- **Code panels.** `renderNP1`/`renderTF1` (steps 1–3) and `renderNPRest`/`renderTFRest` (steps 4–6) build the cells shown in the `<details class="py">` panels (collapsed by default; "Show all code" choice stored in localStorage key `tinycnn-code-open`). `fullScript()` concatenates the non-optional cells for the copy button.
- **Training core.** `makeDataset` (54 shapes), `forward`, `backward` (hand-written backprop), `batchGrads`, `applyUpdate` (SGD / momentum 0.9 / Adam, same formulas as Keras), `runEpoch`, `evaluate`, `randomParams` (Glorot uniform). Seeded with `mulberry32` so "Rerun from same start" is exact.
- **Train UI.** `trainEpoch`, `renderChart` (current and up to 3 earlier runs), `renderTrainStats`, teach step, `pyTrain()` (training script for the current engine, optimizer, learning rate, ReLU and kernel settings).

## Invariants: keep these true

- **Printed outputs must match real Python exactly.** `npm test` enforces it. If you add a code cell with an `out`, it must be checked.
- **Weights are stored rounded to 4 decimals** (`q4`) after every training update, so the Python literals on the page reproduce the page's numbers exactly.
- **Flatten order is channels-first:** x0–x3 from kernel 1, x4–x7 from kernel 2. The TensorFlow code uses `tf.transpose(pool, [0, 3, 1, 2])` and the Keras training model uses `Permute((3, 1, 2))` for that. Plain Keras `Flatten()` interleaves; the page shows this on purpose.
- **Keras Dense kernels are (8, 4)**, the transpose of the page's table. The code transposes with `tf.transpose(W)`.
- Training always runs in plain JavaScript, in both engines; the page says so. Keep that honest.

## Page conventions

- Colours only via the CSS tokens at the top of the file, with light and dark values (both the `prefers-color-scheme` block and `[data-theme]`). No literal colours in components.
- Sentence case for all labels and headings. Plain, direct copy.
- Keep it a single file with no build step. New libraries only from jsDelivr or cdnjs, pinned to an exact version.
- Must work at phone width (about 400px) with no horizontal page scroll.
