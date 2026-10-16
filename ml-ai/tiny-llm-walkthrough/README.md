# Tiny LLM Walkthrough

An interactive page that shows how a language model predicts the next word, small enough to follow every number.

A sentence is split into tokens, each token is looked up in an embedding table, a position encoding is added, one self-attention head mixes information between the words, and the last position is compared with every token's embedding to give a probability for each possible next word. The vocabulary has 8 words and the embeddings have 4 dimensions.

**Open it:** https://jeffrey-e.github.io/learning-tools/ml-ai/tiny-llm-walkthrough/

## What the page shows

- **Tokens.** Build a sentence from the vocabulary and see each word's token id.
- **Embeddings.** An editable embedding table with readable columns (animal, place, action, grammar), and the cosine similarity between every pair of words.
- **Position.** The sinusoidal position encoding, and how it makes two copies of the same word different.
- **Self-attention.** Editable query, key and value matrices, the causal attention weights, and the full calculation for any word.
- **Next word.** Logits, softmax with a temperature control, and generation one token at a time (greedy or sampled).
- **Real code beside each step.** The matching NumPy code with the exact output it prints for the current values.

## Run it locally

Open `index.html` in a browser. It needs `model.js`, `weights.js` and `../../assets/site.css` next to it, so download the whole repository rather than the single file.

```bash
pip install -r requirements.txt
python python/llm_numpy.py
```

## How it's built and verified

| File | What it does |
|---|---|
| `index.html` | The page: layout and interaction. Styles come from the shared `assets/site.css` |
| `model.js` | The model maths and the NumPy code generator, shared by the page and the tests. No DOM access |
| `weights.js` | Default weights. **Generated** by `tools/train_weights.py` |
| `python/llm_numpy.py` | The page's NumPy code for the default sentence. **Generated** by `tools/generate_python.js` |
| `tests/page_smoke.js` | Loads the page headlessly (jsdom) and clicks through every control |
| `tests/dump_states.js`, `tests/check_outputs.py` | Run the page's NumPy code in 54 states in real NumPy and fail on any difference |

```bash
npm ci
pip install -r requirements.txt
npm test             # smoke test + every printed output checked against real NumPy
npm run generate     # after changing model.js or weights.js
npm run train        # retrain the attention matrices (about a minute), then regenerate
```

The embeddings are set by hand so that their columns can be read. The four attention matrices are trained with Adam on four sentences ("the cat sat on the mat ." and its variants) and rounded to 2 decimals. The loss reaches its minimum: "cat or dog" and "sat or ran" are genuinely 50/50 in that corpus, and every other next word is certain.
