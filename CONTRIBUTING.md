# Contributing

How the site is put together, how to add a walkthrough, and how to check a change before it goes live.

## How the site works

- **Static pages, no build step.** GitHub Pages serves the `main` branch root. `index.html` is the home page, `ml-ai/index.html` the section page with one card per walkthrough.
- **One folder per walkthrough**, self-contained: its page, its maths, its tests, its Python, its own `package.json` and `requirements.txt`.
- **One source for the maths.** The page, the Python it shows and the tests use the same JavaScript module (`model.js`, `page.js`) or the same generated data (`data.js`). Python that trains a model writes those files; nobody edits them by hand.
- **Every printed output is checked.** Each code cell on a page comes with the output it prints. The tests run that code in real Python and compare.
- **Honest about what's real.** Each page says where every number comes from (input data, design choice, learned, computed), which values were set by hand, and where a model fails.

## Adding a walkthrough

1. **Create the folder** under a section, e.g. `ml-ai/my-walkthrough/`. Start from the closest existing walkthrough: `tiny-llm-walkthrough` for a small model with hand-checkable numbers, `image-recognition-walkthrough` for a trained Keras model running in TensorFlow.js.
2. **Page.** `index.html` links `../../assets/site.css` and the same Google Fonts link as the other pages, and has a breadcrumb back to the section and home. Put the maths and the code-cell generator in a separate pure module (no DOM access), so the tests can load it in Node.
3. **Tests.** A `package.json` whose `npm test` covers at least: a smoke test that loads the page and clicks through it (jsdom, or Playwright when the page needs a real browser), and a check that runs every code cell in real Python and compares its printed output. Add a `requirements.txt`.
4. **Docs.** A `README.md` for people (what the page shows, how it's built, how to run it) and a short `CLAUDE.md` for Claude Code (commands, generated files, gotchas).
5. **Wire it in.** Add a card to the section's `index.html`, the folder to the matrix in `.github/workflows/verify.yml`, and a row to the table in the root `README.md`. For a new section, also create its folder with an `index.html` (like `ml-ai/index.html`) and add a card for it to the home page.
6. **Third-party content** (datasets, photos, pretrained weights): see [Data and licences](#data-and-licences).

External tools the section recommends go in its "Other useful tools" list: one `<li>` card with an `<h3>` title, a sentence on what it shows, and a `.url` line naming the site and its makers.

## Conventions

- **Colours** only through the CSS tokens in `assets/site.css`, which have light and dark values. No literal colours in components; canvases read the tokens at draw time. `tiny-cnn-walkthrough/index.html` keeps its own inline copy of the styles (it works as a single downloaded file): if you change a token or a shared component, change both.
- **Copy:** sentence case for labels and headings; plain, direct sentences.
- **Layout:** works at about 400px wide with no horizontal page scroll; wide tables and diagrams scroll inside their own box. Check light and dark mode.
- **Libraries** only from jsDelivr or cdnjs, pinned to an exact version, with a Subresource Integrity hash (`integrity="sha384-…"` and `crossorigin="anonymous"`).
- **Generated files** (`weights.js`, `data.js`, `python/`, `notebooks/`, `model/`, `images/`) are never edited by hand: change the generator and rerun it. Each walkthrough's README says which command regenerates what.
- **Code panels** use the shared `details.py` component ("Show code"); the "Show all code" choice is remembered in `localStorage` under a key named after the page.

## Verifying a change

- Run `npm test` in the walkthrough's folder. It must end with "all checks passed"; a change isn't done until it does.
- For visible changes, look at the page at about 400px and at full width, in light and dark mode (`python3 -m http.server` from the repository root).
- CI runs every walkthrough's tests on every push (`.github/workflows/verify.yml`). It runs on x86 Linux; the dev container on a Mac runs on ARM. A float printed on a rounding boundary can differ in its last digit between the two: the image recognition checker allows exactly that and lists such cases; nothing else is relaxed.

## Data and licences

- **Credit everything you didn't make:** author, source, licence, and what was changed (e.g. "resized to 224×224"). Show it on the page and keep it in the repository (a `data/README.md` or `LICENSES.md` in the walkthrough).
- **Use only licences that allow republishing:** e.g. CC0, public domain, CC BY, CC BY-SA, Apache 2.0, MIT. CC BY-SA content stays CC BY-SA when modified.
- **List it at the end of the root `LICENSE`**, which says what the MIT licence does not cover.
- **Pin and check downloads** that code depends on (e.g. pretrained weights: a SHA-256 in the code, checked before use).

## Workflow

- Work on `main`; GitHub Pages publishes it. Changes are reviewed in VS Code before they're committed.
- Commit messages: a short subject in the style of the history ("Adding image recognition", "Fixing CI issues").

## Development environment

The repository is developed in a dev container (`/workspaces/Jeffrey/.devcontainer/`, outside this repository) on an ARM Mac: Node 24 (as in CI), Python 3.11 with TensorFlow in the user's site-packages.

- Run Python tools that read downloaded files with `python3 -P` (keeps the script's folder off the import path). `python3 -I` would also drop the user site-packages, and with them TensorFlow.
- Converting a Keras model to TensorFlow.js needs its own environment: see `ml-ai/image-recognition-walkthrough/requirements-export.txt`.
- If VS Code's git reports "dubious ownership", the repository needs to be marked as safe (`git config --global --add safe.directory <path>`); the dev container's `post-create.sh` does this on every build.
