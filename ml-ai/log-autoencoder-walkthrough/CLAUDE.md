# CLAUDE.md

Guidance for Claude Code working on the Log autoencoder walkthrough. See also the repository-level `CLAUDE.md`.

## What this is

A teaching page for anomaly detection in SSH logs with a character-level autoencoder: Elastic's public `auth.log` (sshd lines, prefix stripped), rule-based labels (legit / attack / other), training on legitimate lines only, characters to ids (128 per line, 0 = padding, 1 = unknown), `Embedding(V, 8)`, `Conv1D(16, 5)` + pool, `Conv1D(4, 5)` + pool (32×4 bottleneck), upsampling + `Conv1D` back to 128 positions, softmax per character, score = mean −log p(true character), threshold = 99th percentile of validation scores. Lives in `ml-ai/log-autoencoder-walkthrough/`. There is no build step.

## Layout

- `index.html`: markup, page-specific CSS (tokens only) and the UI script. Shared styles come from `../../assets/site.css`.
- `model.js`: the forward pass (`run`, `conv1d`, `maxpool`, `upsample`), `RULES`/`label`, `detect`, and the Keras code generator (`pythonCells`, `fullScript`). Loadable in the browser (`LogAE` global) and in Node. Keep DOM code out of it.
- `tools/train_model.py`: the real Keras training. **Generates** `weights.js`, `python/weights.json` and `data.js`. Don't edit those by hand.
- `python/autoencoder_tf.py`: **generated** by `tools/generate_python.js`.
- `tests/`: `page_smoke.js` (jsdom), `dump_states.js` (also checks `model.js` against every score in `data.js`), `check_outputs.py` (runs the Keras cells in real TensorFlow), `states_config.js` (the lines tested; `DEFAULT_LINE`).

## Commands

```bash
npm ci && pip install -r requirements.txt
npm test             # must end with "all checks passed"
npm run train        # retrain and regenerate everything (deterministic on CPU)
npm run generate     # after changing model.js
```

## Invariants: keep these true

- **Printed outputs must match real TensorFlow exactly.** Computed values print with 3 decimals and weights with 4, which absorbs float32 vs float64 differences. Any new cell needs an `out` built in `pythonCells`.
- **One source for the maths.** Keras (`train_model.py`, the Python cells) and `model.js` must compute the same thing: same layer order and names, "same" padding, score clipped at p = 1e-7, mean over real characters. `dump_states.js` fails if any score differs from `data.js` by more than 1e-3.
- **`RULES` exist twice** (`model.js` and `tools/train_model.py`); the tests compare the labels.
- **Count characters as code points** (`Array.from` in JavaScript), as Python does.
- **Be honest about what's real.** Labels come from rules, not from the model; the page says so, and it shows the false alarms and the brute-force success the model can't see.
- `data/auth.log` stays unmodified, with its attribution in `data/README.md`.

## Page conventions

Same as the rest of the site: colours only through the CSS tokens in `assets/site.css`, sentence case, plain copy, works at about 400px wide with no horizontal page scroll, no new libraries unless pinned from jsDelivr or cdnjs.
