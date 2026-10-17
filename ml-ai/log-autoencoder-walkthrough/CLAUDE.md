# Log autoencoder walkthrough

Anomaly detection in Elastic's public SSH `auth.log` with a Conv1D autoencoder: four tokenizers, each with its own model, and a filter for pasted lines. The model, data and files: `README.md`.

## Commands

- `npm test`
- `npm run train` retrains all four models (about 1.5 minutes, deterministic on CPU), then regenerates.
- `npm run generate` after changing `model.js`.

## Generated: never edit by hand

- `weights.js`, `data.js`, `python/weights_*.json` (`tools/train_model.py`); `python/autoencoder_tf.py` (`tools/generate_python.js`).
- `data/auth.log` stays unmodified (attribution in `data/README.md`).

## Gotchas

- `model.js` is the one source for the maths: pure, no DOM. Every function takes one tokenizer's model (`AE_WEIGHTS[name]`).
- The labelling `RULES` and the tokenizers exist three times: in `model.js`, in the Python it writes into the cells, and in `tools/train_model.py`. Keep the regexes identical, and use `[^\n]`, not `.`, so both languages match the same characters.
- Count characters as code points (`Array.from`, the `u` flag), as Python does.
- Scores in `data.js` keep 7 significant digits (the words and masked models score legitimate lines around 0.0001). Thresholds have 3 significant digits and are at least 0.0001, so both languages print them the same way.
- `dump_states.js` fails if `model.js` and Keras disagree on any score by more than 1e-3.
- The batch filter outputs the original lines, never reconstructions: a score belongs to the line in the same position. The `batch` cell covers the selected tokenizer only; the any/all vote runs on the page, and the code panel says so.
- Labels come from rules, not from the model. The page says so, and shows the false alarms and the brute-force success it can't catch: keep that honest.
