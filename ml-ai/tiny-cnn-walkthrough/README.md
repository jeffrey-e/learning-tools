# Tiny CNN Walkthrough

An interactive, single-page explanation of how a convolutional neural network works, small enough to follow every number.

A 6×6 drawing goes through two 3×3 kernels, ReLU, 2×2 max pooling, flatten, a dense layer and softmax, and comes out as one of four classes: vertical line, horizontal line, cross/box or empty.

**Open it:** https://jeffrey-e.github.io/learning-tools/ml-ai/tiny-cnn-walkthrough/

**Run the Python in your browser:** [Open the notebook in Google Colab](https://colab.research.google.com/github/jeffrey-e/learning-tools/blob/main/ml-ai/tiny-cnn-walkthrough/notebooks/tiny_cnn_walkthrough.ipynb)

## What the page shows

- **Every stage, step by step.** Watch the 3×3 window slide over the image, hover any number to see which input pixels produced it, and follow the values through ReLU, pooling, flatten, the dense layer and softmax.
- **Everything is editable.** Draw your own image, and change any kernel value, dense weight or bias.
- **Where every value comes from.** Each number is tagged as input data, a design choice, a learned parameter or a computed value.
- **Real code beside each step.** Matching NumPy or TensorFlow code with the exact output it prints for the current values. With the TensorFlow engine, the page itself computes with TensorFlow.js.
- **Train mode.** Start from random weights and watch gradient descent find them, compare SGD, SGD + momentum and Adam from the same starting point, or take a single gradient step on your own drawing.

## Run it locally

The page is one self-contained file: download `index.html` and open it in a browser. It loads TensorFlow.js and web fonts from public CDNs. Offline, everything except the TensorFlow engine still works.

For the Python scripts in `python/`:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt    # NumPy only? pip install numpy
python python/cnn_numpy.py
```

| File | What it does |
|---|---|
| `python/cnn_numpy.py` | Forward pass in NumPy, with the page's default image and hand-set weights |
| `python/cnn_tensorflow.py` | The same forward pass with TensorFlow ops and a Keras `Dense` layer |
| `python/train_numpy.py` | Training from random weights with hand-written backpropagation |
| `python/train_tensorflow.py` | Training with Keras `model.fit` and Adam |
| `notebooks/tiny_cnn_walkthrough.ipynb` | All of the above as a step-by-step notebook |

The page's **Copy full script** and **Copy training script** buttons give a version with your own drawing, weights and settings.

## How the outputs are verified

Every printed output the page shows is checked against real NumPy and TensorFlow by running the generated code. The checks run on every push (see `.github/workflows/verify.yml`), or locally:

```bash
npm ci
pip install -r requirements.txt
npm test
```

TensorFlow computes in float32, so the generated code limits printouts to 4 decimals (`np.set_printoptions(precision=4, suppress=True)`). That keeps the output identical across machines.

## Repository layout

```
index.html            the whole interactive page (HTML, CSS and JavaScript in one file)
python/               standalone scripts generated from the page
notebooks/            Colab / Jupyter notebook generated from the page
tests/                headless page driver + Python output checker
tools/                regenerates python/ and the notebook from index.html (npm run generate)
```
