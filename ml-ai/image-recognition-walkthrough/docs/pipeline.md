# The data and model pipeline

How the photos, labels, models and generated files of this walkthrough are made, and how to rebuild them. Every step is seeded and deterministic on a CPU: running a step again on unchanged input gives the same output. Run every command from this folder.

## Overview

| Step | Command | Time | Reads | Writes |
|---|---|---|---|---|
| 1. Find candidates | `python3 -P tools/fetch_commons.py` | ~7 min | Wikimedia Commons | `data/candidates.json`, `data/candidates/` (not committed) |
| 2. First pass | `python3 -P tools/first_pass.py` | ~10 s | candidates | `data/labels.json` (`by: "first-pass"`) |
| 3. Human review | `python3 -P tools/review_server.py` | you | candidates, labels | `data/labels.json` (`by: "user"`) |
| 4. Prepare the dataset | `python3 -P tools/prepare_data.py` | ~10 s | labels, candidates, Commons metadata | `images/`, `data/photos.json` |
| 5. Tuning experiments | `python3 -P tools/experiments.py` | ~30 min | photos | `data/experiments.json` |
| 6. Train the final models | `python3 -P tools/train.py` | ~3 min | photos, experiments | `data.js`, `python/*.weights.h5`, `python/export_*` |
| 7. Convert to TensorFlow.js | `.venv-export/bin/python tools/export_tfjs.py` | ~10 s | `python/export_*`, `data.js` | `model/scratch/`, `model/transfer/` |
| 8. Regenerate the script | `npm run generate` | 1 s | `page.js`, `data.js` | `python/walkthrough.py` |
| 9. Verify | `npm test` | ~1 min | everything | nothing (fails on any difference) |

**What to rerun when:**
- **A label changed:** steps 4 to 9. The split can change, so the experiments must run again.
- **An experiment or training setting changed:** steps 5 (or only 6, if the choice of run doesn't change) to 9.
- **Only `page.js` or `index.html` changed:** steps 8 and 9.

## 1. Find candidates

`fetch_commons.py` collects files from the breed categories on Wikimedia Commons (Siberian Husky and its subcategories, Alaskan Malamute, heads, puppies) and a search for "Alaskan Malamute". It keeps only freely licensed files (CC0, public domain, CC BY, CC BY-SA), stores each file's author, licence and page in `data/candidates.json`, and downloads a 500px thumbnail to `data/candidates/`. The label is a first guess: the breed a file was found under.

## 2. First pass

`first_pass.py` drops photos in which a pretrained ImageNet MobileNetV2 sees no dog at all (no dog class in its top 5). It does **not** decide breeds, so the labels don't come from the kind of model the walkthrough tests. A visual pass over contact sheets then dropped drawings, other breeds, sled teams of mixed-breed "Alaskan huskies", photos with several dogs or people, historical black-and-white photos and close-ups; those decisions are in `data/labels.json` with a `note`.

## 3. Human review

```bash
python3 -P tools/review_server.py      # then open http://localhost:8001/
```

Every candidate with its proposed label. Set each to husky, malamute or drop (keys h, m, d), then **Save**: the server writes `data/labels.json`. Labels set here get `by: "user"` and are never overwritten by step 2. To check a few photos only: `http://localhost:8001/?ids=c0077,c0098`. The server listens on localhost only.

## 4. Prepare the dataset

`prepare_data.py` keeps the photos labelled husky or malamute and:
- **groups them by session** (the same uploader on the same day, from the Commons metadata), so that a group never spans two splits and the test set can't contain another shot of a training dog;
- **splits each breed by group**: about 20% test, 15% validation, the rest training;
- **pads each photo to a square** with grey (nothing is cut off) and resizes it to 224×224, into `images/`.

## 5. Tuning experiments

`experiments.py` trains every variant listed in `A_RUNS` and `B_RUNS` and records its training and validation accuracy, plain and balanced (the mean of the husky and malamute recall). It picks, for each model, the run with the best balanced validation accuracy. The test photos are not used here. The page's step 7 shows the table.

## 6. Train the final models

`train.py` trains exactly the chosen runs, through the same functions as the experiments, and evaluates them once on the test photos. It predicts one photo at a time and stores 7 significant digits, because the page's code cells predict one photo at a time. It writes `data.js` (photos and credits, predictions, ImageNet's top 3, curves, metrics, experiments, the models' structure and class activation maps for the test photos) and the weights for the Python cells and the export.

Pretrained weights are downloaded through `tools/pretrained.py`, which checks them against pinned SHA-256 hashes first.

## 7. Convert to TensorFlow.js

The TensorFlow.js converter needs TensorFlow 2.15 with `tf_keras`, so it runs in its own environment:

```bash
python3 -m venv .venv-export
.venv-export/bin/pip install -r requirements-export.txt
.venv-export/bin/python tools/export_tfjs.py
```

`export_tfjs.py` rebuilds both models with the extra outputs the page shows (early and late feature maps), loads the trained weights, checks that its predictions match `data.js`, and writes the TF.js layers format: model A in float32, model B in float16 (half the download, about 0.001 off). It imports only the converter's Keras module, because the full `tensorflowjs` package imports optional dependencies (JAX, TensorFlow Decision Forests) that don't install on every machine.

## 8 and 9. Regenerate and verify

`npm run generate` writes `python/walkthrough.py`. `npm test` runs:
- `tests/page_smoke.js`: the page in headless Chromium (Playwright) at 400px and full width, with the models loading, every step clicked through, and the code panels compared with `page.js`;
- `tests/dump_states.js` and `tests/check_outputs.py`: every code cell run in real Keras for a few test photos, compared with the page's output;
- `tests/check_tfjs.js`: the TF.js models against Keras on the same pixels.

Install the browser once with `npx playwright install chromium`.
