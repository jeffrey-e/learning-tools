# CLAUDE.md

Guidance for Claude Code working on the Image recognition walkthrough. See also the repository-level `CLAUDE.md`.

## What this is

A teaching page on image recognition: can a model tell a Siberian Husky from an Alaskan Malamute? About 400 Wikimedia Commons photos (labels reviewed by a person; the exact counts are in `data.js`), model A (small CNN from scratch, Keras) vs model B (frozen MobileNetV2 + trained head), tuning experiments, class activation maps, mistakes, and your own photo. Both models run in the browser with TensorFlow.js 4.22.0 (jsDelivr, pinned, with SRI). Lives in `ml-ai/image-recognition-walkthrough/`; no build step.

## Layout

- `index.html`: markup, page CSS (tokens only), UI script. Loads `data.js`, `page.js`, `model/*/model.json`.
- `page.js`: the Keras cells (`pythonCells`) with outputs built from `data.js`, `fullScript`, `cam`. Pure; browser global `ImgPage`.
- Generated, never edited by hand: `images/`, `data/photos.json` (`prepare_data.py`), `data/experiments.json` (`experiments.py`), `data.js` and `python/*.weights.h5`, `python/export_*` (`train.py`), `model/` (`export_tfjs.py`), `python/walkthrough.py` (`generate_python.js`).
- `data/labels.json`: the reviewed labels (`by: "user"` entries come from the human review and are never overwritten by tools). `data/candidates/` (raw downloads) is git-ignored.

## Commands

```bash
npm ci && npx playwright install chromium && pip install -r requirements.txt
npm test          # must end with "all checks passed" and the TF.js line
npm run generate  # after changing page.js
```
The pipeline order and the export environment are in `README.md` and `requirements-export.txt`.

## Invariants: keep these true

- **Printed outputs must match real Keras.** `train.py` predicts one photo at a time and stores 7 significant digits, because the cells predict one photo at a time; batched predictions can flip a rounding. `check_outputs.py` accepts a decimal number that is one off in its last printed digit (x86 with oneDNN and ARM round differently; such cases are listed), never a different whole number or text. It downloads the pretrained files before running cells, so no download message lands in an output.
- **The test photos are used once.** Choices are made on validation (`experiments.py`, balanced accuracy); `train.py` builds exactly the chosen runs through the same functions.
- **Sessions never span splits** (author + day), so the test can't contain another shot of a training dog.
- **Pretrained weights are verified** against the SHA-256 in `tools/pretrained.py` before any tool uses them. The TF.js script tag keeps its `integrity` hash; `page_smoke.js` fails if it doesn't match the pinned npm copy.
- **Be honest:** model A's poor result is shown as it is; labels come from people, and the page says some "mistakes" can be label errors.
- Credits for every photo (author, licence, Commons page) stay on the page and in `data/photos.json`. `LICENSES.md` (and the note at the end of the root `LICENSE`) says which parts are not MIT; update it if third-party content changes.

## Page conventions

Same as the rest of the site: colours only through the CSS tokens in `assets/site.css` (canvases read them at draw time), sentence case, plain copy, about 400px wide with no horizontal page scroll.
