# Licences in this folder

The repository's MIT licence (`../../LICENSE`) covers the code and text written for this walkthrough. Two parts come from others and keep their own licences:

## Photos: `images/`

Every photo comes from [Wikimedia Commons](https://commons.wikimedia.org/) and keeps the licence of its original: CC0, public domain, CC BY (2.0 to 4.0) or CC BY-SA (2.0 to 4.0). They are **not** under the MIT licence.

- **Who made each photo, and under which licence:** `data/photos.json` (fields `author`, `licence`, `licence_url`, `page`), and the credits list on the page (step 1).
- **What was changed:** each photo was padded to a square with grey and resized to 224×224 pixels. Nothing else.
- **CC BY-SA photos:** these modified versions are shared under the same licence as their original, as CC BY-SA requires.

The trained weights in `python/scratch.weights.h5` and `model/scratch/` (model A) and `python/transfer_head.weights.h5` (model B's head) were learned from these photos. They contain no photos.

## The pretrained model: `model/transfer/`

Model B builds on **MobileNetV2** (Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov and Liang-Chieh Chen, Google, 2018, [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)), with the ImageNet weights from [Keras Applications](https://keras.io/api/applications/mobilenet/), licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

- **What was changed:** the pretrained weights were combined with the head trained here, converted to the TensorFlow.js format, and stored as 16-bit floats (`tools/export_tfjs.py`).
- The weight files are checked against pinned SHA-256 hashes before use (`tools/pretrained.py`).
- The weights were trained by Google on ImageNet, whose photos are licensed for non-commercial research. Models trained on ImageNet are widely published; check your own situation before using them commercially.

## Loaded from elsewhere

- **TensorFlow.js 4.22.0**, Google, Apache License 2.0: loaded by the page from jsDelivr, not stored here.
