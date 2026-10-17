# Learning tools

Interactive pages that explain machine learning by following real examples through every step, small enough to see every number. Each step shows the matching Python code with the exact output it prints, checked against real NumPy, TensorFlow or Keras on every push.

**Open the site:** https://jeffrey-e.github.io/learning-tools/

## The walkthroughs

| Walkthrough | What it shows |
|---|---|
| [Tiny CNN](ml-ai/tiny-cnn-walkthrough/) | One 6×6 image through convolution, ReLU, pooling, a dense layer and softmax; edit any weight and train the network yourself. |
| [Tiny LLM](ml-ai/tiny-llm-walkthrough/) | A sentence through tokens, byte pair encoding, embeddings, position encoding and one self-attention head to a prediction of the next word. |
| [Log autoencoder](ml-ai/log-autoencoder-walkthrough/) | Anomaly detection in a real SSH `auth.log` with a Conv1D autoencoder: four tokenizers, a threshold, and a filter for your own lines. |
| [Image recognition](ml-ai/image-recognition-walkthrough/) | Husky or malamute? A CNN learned from scratch against transfer learning on MobileNetV2, with every tuning experiment, heatmaps of what each model looks at, and your own photo. |

Each folder has a `README.md` that explains how that walkthrough works and is built.

## Run it locally

There is no build step. Serve the repository root (pages load the shared `assets/site.css` with a relative path), then open a walkthrough:

```bash
python3 -m http.server 8000
# open http://localhost:8000/
```

## How the pages are verified

Every walkthrough has its own tests (`npm test` in its folder): they open the page headlessly, click through it, and run every code cell the page shows in real Python, failing on any difference from the output the page prints. GitHub Actions runs all of them on every push and pull request (`.github/workflows/verify.yml`); GitHub Pages serves the `main` branch.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): how to add a walkthrough, the conventions every page follows, and how to verify a change.

## Licence

The code and text are MIT ([LICENSE](LICENSE)). Some folders contain work by others under its own licence, such as the photos and pretrained weights of the image recognition walkthrough and the `auth.log` of the log autoencoder; the end of [LICENSE](LICENSE) lists them.
