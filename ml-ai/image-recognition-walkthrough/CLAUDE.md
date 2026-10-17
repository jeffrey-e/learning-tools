# Image recognition walkthrough

Husky or malamute: model A (a small CNN trained from scratch) against model B (frozen MobileNetV2 + a trained head), both running live in TensorFlow.js. The page and files: `README.md`. The data and model pipeline, step by step: `docs/pipeline.md`.

## Commands

- `npm test` (Playwright: run `npx playwright install chromium` once). `npm run test:quick` skips the training cells.
- `npm run generate` after changing `page.js`.
- Rebuilding data or models is a multi-step pipeline (about 35 minutes); follow `docs/pipeline.md` and rerun only the steps it lists for your change.

## Generated: never edit by hand

- `images/`, `data/photos.json`, `data/experiments.json`, `data.js`, `python/*.weights.h5`, `python/export_*`, `model/`, `python/walkthrough.py`.
- `data/labels.json` holds the human review: entries with `by: "user"` are never overwritten by tools. `data/candidates/` is git-ignored.

## Gotchas

- `train.py` predicts one photo at a time and stores 7 significant digits: the cells predict one photo at a time, and batched predictions can flip a rounding.
- `check_outputs.py` accepts a decimal one off in its last printed digit (x86 and ARM round differently), never a different whole number or text. It downloads the pretrained files before running cells, so no download message lands in an output.
- Settings are chosen on validation photos (`experiments.py`, balanced accuracy). The test photos are used once, in `train.py`, which builds exactly the chosen runs through `experiments.py`'s functions.
- Photo sessions (same author, same day) never span two splits.
- Pretrained weights are checked against the SHA-256 in `tools/pretrained.py`. The TF.js script tag keeps its `integrity` hash; `page_smoke.js` fails if it doesn't match the npm copy.
- The TF.js converter runs in its own environment (`requirements-export.txt`) and imports only its Keras module.
- Retrained models keep their file names, so the page loads them with `cache: 'no-cache'` and retries with `'reload'`.
- Step 5 draws `data.js`'s `arch` (grouped by `train.py`'s `block_name`, the same rules as the `arch` cell) and cuts the TF.js model at a block's last layer for its live output.
- Model A's weak result is shown as it is, and the page says some "mistakes" may be label errors: keep that honest. Third-party credits: `LICENSES.md`.
