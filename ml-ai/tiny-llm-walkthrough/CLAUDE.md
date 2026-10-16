# CLAUDE.md

Guidance for Claude Code working on the Tiny LLM walkthrough. See also the repository-level `CLAUDE.md`.

## What this is

A teaching page that walks a sentence through a tiny decoder-only transformer: tokens (8-word vocabulary; free text is lowercased and split into words and punctuation, unknown pieces are skipped), embeddings (4 dimensions), sinusoidal position encoding, one causal self-attention head, a residual connection, tied output embeddings, softmax with temperature, and generation. Lives in `ml-ai/tiny-llm-walkthrough/` and is served by GitHub Pages from the `main` branch. There is no build step.

## Layout

- `index.html`: markup, page-specific CSS (a few rules, tokens only) and the UI script. Shared styles come from `../../assets/site.css`.
- `model.js`: all maths (`run`, `positional`, `cosine`, `tokenize`) and the NumPy code generator (`pythonCells`, `fullScript`). Pure functions, loadable in the browser (`TinyLLM` global) and in Node (`require`). Keep DOM code out of it.
- `weights.js`: **generated** by `tools/train_weights.py`. Don't edit by hand.
- `python/llm_numpy.py`: **generated** by `tools/generate_python.js`. Don't edit by hand.
- `tests/`: `page_smoke.js` (jsdom, clicks through the page), `dump_states.js` + `check_outputs.py` (runs the generated NumPy in 90 states and compares printed output exactly and arrays to 1e-9).

## Commands

```bash
npm ci && pip install -r requirements.txt
npm test             # must end with "all checks passed"
npm run generate     # after changing model.js or weights.js
npm run train        # retrain attention matrices, then regenerate
```

## Invariants: keep these true

- **Printed outputs must match real NumPy exactly.** Any new code cell needs an `out` built in `pythonCells`, and `npm test` checks it.
- **One source for the maths.** The page, the generated Python and the tests all go through `model.js`. Change the maths there and in the Python it emits, together; `tools/train_weights.py` has its own NumPy `forward` that must stay equivalent.
- **Weights are rounded to 2 decimals**, so the Python literals reproduce the page's numbers exactly.
- **Be honest about what's real.** The embeddings are hand-set and the page says so; the attention matrices are trained.

## Page conventions

Same as the rest of the site: colours only through the CSS tokens in `assets/site.css`, sentence case, plain copy, works at about 400px wide with no horizontal page scroll, no new libraries unless pinned from jsDelivr or cdnjs.
