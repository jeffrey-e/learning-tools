# Log Autoencoder Walkthrough

An interactive page that shows how an autoencoder finds anomalies in an SSH `auth.log`, one step at a time.

An autoencoder (Keras `Embedding` and `Conv1D` layers) is trained only on legitimate sshd messages, with a choice of four tokenizers: characters, masked characters (IP addresses and numbers replaced by `<ip>` and `<num>`), words with the same placeholders, and subwords by byte pair encoding. Each tokenizer has its own trained model. It learns to rebuild them through a narrow bottleneck; lines it can't rebuild well get a high score and are flagged as anomalous.

**Open it:** https://jeffrey-e.github.io/learning-tools/ml-ai/log-autoencoder-walkthrough/

## What the page shows

- **The data.** A real `auth.log` from Elastic's examples (Apache-2.0), filtered to sshd and stripped of date, host and process id.
- **Labels, balance and split.** Rule-based labels, and why the data's balance (72% attack lines) means the model is trained on legitimate lines only.
- **Tokens to ids, embedding, encoder, decoder.** Switch the tokenizer in the toolbar and every step follows. Every intermediate array for the selected line is shown as a heat strip: embedding, both Conv1D layers, the 32×4 bottleneck, the decoder and its per-character probabilities.
- **Reconstruction and error.** The line and its reconstruction, aligned character by character, with the error per character and the line's score.
- **Training.** The real training and validation loss curve.
- **Detection results.** A threshold slider (logarithmic), score distributions, precision and recall, a table comparing the four tokenizers, and lists of flagged lines, legitimate lines after reconstruction, false alarms, missed attacks and a case a line-by-line model can't catch.
- **Filter your own lines.** Paste lines from any `auth.log` (or load a random sample): each line is scored on its own, lines scored as normal are removed, and the original anomalous lines remain, with a vote across the four tokenizer models and a check against the labels for lines that are in this log.
- **Real code beside each step.** The matching TensorFlow/Keras code with the exact output it prints.

## The model

- **Data:** the sshd lines of Elastic's `auth.log`, with the date, host and process id stripped. Labels come from rules on the message type (legitimate, attack, other); the model never sees them. Only legitimate lines are used for training: 60% train, 20% validation, 20% test, the same split for every tokenizer.
- **Tokens to ids:** every line becomes a fixed number of ids L (0 = padding, 1 = unknown).

  | Tokenizer | Tokens | L |
  |---|---|---|
  | `chars` | every character | 128 |
  | `masked` | characters, with IP addresses and numbers replaced by `<ip>` and `<num>` | 112 |
  | `words` | words and punctuation, same placeholders | 32 |
  | `bpe` | subwords from 200 byte pair encoding merges learned on the training lines | 64 |

- **Autoencoder:** `Embedding(V, 8)` → `Conv1D(16, 5)` + max pooling → `Conv1D(4, 5)` + max pooling (the bottleneck: L/4 positions × 4 numbers) → upsampling and `Conv1D` back to L positions → a softmax over the vocabulary at every position. Each tokenizer has its own model.
- **Score:** the mean of −log p(true token) over the real tokens. **Threshold:** the 99th percentile of the validation scores.

## Run it locally

Serve the repository root (the page loads `weights.js`, `data.js`, `model.js` and `../../assets/site.css`), for example `python3 -m http.server` from the repository root, then open `/ml-ai/log-autoencoder-walkthrough/`.

```bash
pip install -r requirements.txt
python python/autoencoder_tf.py        # run from this folder
```

## How it's built and verified

| File | What it does |
|---|---|
| `index.html` | The page: layout and interaction. Styles come from the shared `assets/site.css` |
| `model.js` | The forward pass in plain JavaScript, the labelling rules and the Keras code generator. No DOM access |
| `tools/train_model.py` | Labels and splits the lines, learns the BPE merges, trains one model per tokenizer with Keras; writes `weights.js`, `python/weights_<tokenizer>.json` and `data.js` |
| `weights.js`, `python/weights_*.json` | Trained weights per tokenizer, rounded to 4 decimals, plus the loss curve (and the BPE merges). **Generated** |
| `data.js` | Every sshd message with its label, split and a score per tokenizer. **Generated** |
| `python/autoencoder_tf.py` | The page's Keras code for the default line, character tokenizer. **Generated** by `tools/generate_python.js` |
| `data/auth.log` | The source log, unmodified (see `data/README.md`) |
| `tests/page_smoke.js` | Loads the page headlessly (jsdom) and clicks through every control |
| `tests/dump_states.js`, `tests/check_outputs.py` | Check that `model.js` reproduces every score Keras computed, and run the page's code in real TensorFlow for every tokenizer, failing on any difference |

```bash
npm ci
pip install -r requirements.txt
npm test             # smoke test + every printed output checked against real TensorFlow
npm run train        # retrain all four models (about 1.5 minutes on a CPU, deterministic), then regenerate
npm run generate     # after changing model.js
```
