# Image Recognition Walkthrough

An interactive page about how image recognition works, on a hard case: telling a Siberian Husky from an Alaskan Malamute.

Real photos go through two models: model A, a small CNN learned from scratch on our 261 training photos, and model B, transfer learning on MobileNetV2 pretrained on ImageNet. The page shows a photo as numbers, the first-layer filters and their feature maps, deeper layers, both whole models side by side (click any block to see its live output), both models' training, every tuning experiment, what each model looks at (class activation maps), its mistakes, and lets you try your own photo. Both models run in the browser with TensorFlow.js.

**Open it:** https://jeffrey-e.github.io/learning-tools/ml-ai/image-recognition-walkthrough/

## The data

403 photos from [Wikimedia Commons](https://commons.wikimedia.org/) (CC0, public domain, CC BY, CC BY-SA), 303 huskies and 100 malamutes, resized and padded to 224×224. Every photo's author, licence and Commons page are listed on the page (Credits, under step 1) and in `data/photos.json`.

How the labels were made: 793 candidates from the breed categories and a search, photos without any dog dropped by a pretrained model (which did not decide breeds), a visual first pass, and a final human review of every label. The steps are in [docs/pipeline.md](docs/pipeline.md). Photos of one author on one day stay in the same split, so the test set can't contain another shot of a training dog.

## The models

| | Model A: from scratch | Model B: transfer learning |
|---|---|---|
| Layers | 4 conv layers, average pooling, dense | MobileNetV2 (frozen) + dropout + dense |
| Learned from our photos | 98,418 weights | 2,562 weights |
| Test accuracy (balanced) | 51% (52%) | 89% (86%) |

Model B builds on MobileNetV2 ([Sandler et al., Google, 2018](https://arxiv.org/abs/1801.04381)) with the ImageNet weights from [Keras Applications](https://keras.io/api/applications/mobilenet/), Apache License 2.0. The tools check the downloaded weight files against pinned SHA-256 hashes (`tools/pretrained.py`); the files are HDF5 arrays, no code. `model/transfer/` serves them converted to TensorFlow.js (float16). ImageNet's photos themselves are licensed for research use. TensorFlow.js 4.22.0 is loaded from jsDelivr with Subresource Integrity.

## Licences

The code and text are MIT, like the rest of the repository. The photos in `images/` keep their own Commons licences, and `model/transfer/` is Apache 2.0 (MobileNetV2): see [LICENSES.md](LICENSES.md).

## Files

| File | What it does |
|---|---|
| `index.html` | The page. Styles from the shared `assets/site.css`; TF.js runs both models live |
| `page.js` | The Keras code shown beside each step, with the exact output it prints. No DOM access |
| `data.js` | Photos, credits, both models' Keras predictions, curves, experiments. **Generated** by `tools/train.py` |
| `model/scratch/`, `model/transfer/` | The models in TF.js format. **Generated** by `tools/export_tfjs.py` |
| `images/` | The 403 photos, 224×224. **Generated** by `tools/prepare_data.py` |
| `python/` | Trained weights for the Python cells, and `walkthrough.py` (**generated** by `tools/generate_python.js`) |
| `tools/` | The pipeline, in order: `fetch_commons.py`, `first_pass.py`, review, `prepare_data.py`, `experiments.py`, `train.py`, `export_tfjs.py` |
| `tests/` | `page_smoke.js` (headless Chromium), `check_outputs.py` (every printed output against real Keras), `check_tfjs.js` (TF.js models against Keras) |

## Run it

```bash
npm ci                      # Playwright and TF.js for the tests
npx playwright install chromium
pip install -r requirements.txt
npm test                    # browser smoke test + every output checked against real Keras and TF.js
python python/walkthrough.py
```

Rebuilding the data or the models is a multi-step pipeline (about 35 minutes): see [docs/pipeline.md](docs/pipeline.md).
