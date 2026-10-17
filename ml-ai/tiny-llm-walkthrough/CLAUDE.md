# Tiny LLM walkthrough

A sentence through a tiny decoder-only transformer (8-word vocabulary, 4 dimensions, one attention head), plus a byte pair encoding side step. How it works and is built: `README.md`.

## Commands

- `npm test`
- `npm run generate` after changing `model.js` or `weights.js`.
- `npm run train` retrains the attention matrices (about a minute, deterministic), then regenerates.

## Generated: never edit by hand

- `weights.js` (`tools/train_weights.py`), `python/llm_numpy.py` (`tools/generate_python.js`).

## Gotchas

- All maths lives in `model.js` (pure, loads in the browser and in Node): keep DOM code out of it.
- Change the maths in `model.js` and the Python it emits together. `tools/train_weights.py` has its own NumPy `forward` that must stay equivalent.
- BPE ties go to the pair seen first, in both languages (JavaScript `Map` order with a strict `>`, Python `dict` order with `max`): keep them in step or the merge lists diverge.
- Weights are rounded to 2 decimals, so the Python literals reproduce the page's numbers exactly.
- The embeddings are set by hand and the attention matrices are trained; the page says so. Keep that honest.
