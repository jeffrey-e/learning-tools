# Tiny CNN walkthrough

One 6×6 image through a tiny CNN, with editable weights, NumPy and TensorFlow code cells, and a train mode. How it works, and how `index.html`'s script is organised: `README.md`.

## Commands

- `npm test`: about a minute, mostly TensorFlow. `npm run test:quick` skips TensorFlow training.
- `npm run generate` after changing `index.html` (regenerates `python/` and the notebook). The notebook build needs `pip install nbformat nbclient ipykernel`.

## Generated: never edit by hand

- `python/`, `notebooks/` (from `index.html`, via `tools/`).

## Gotchas

- The page is one self-contained file (CSS, HTML, JS inline) with its own copy of the site styles: keep it in sync with `assets/site.css`.
- New code cells print through `npStr`/`decStr`/`pyRound4`, which reproduce NumPy's printing at 4 decimals; their `out` is checked against real Python.
- Weights are rounded to 4 decimals (`q4`) after every training update, so the Python literals reproduce the page exactly.
- Flatten order is channels-first (x0–x3 from kernel 1, x4–x7 from kernel 2): the code uses `tf.transpose(pool, [0, 3, 1, 2])` and Keras `Permute((3, 1, 2))`. Plain `Flatten()` interleaves; the page shows this on purpose.
- Keras Dense kernels are (8, 4), the transpose of the page's table: the code uses `tf.transpose(W)`.
- Training always runs in plain JavaScript, in both engines; the page says so. Keep that honest.
