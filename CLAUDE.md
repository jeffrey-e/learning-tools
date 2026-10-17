# learning-tools

Static site of interactive ML walkthroughs, served by GitHub Pages from `main`, no build step. Each folder under `ml-ai/` is a self-contained walkthrough with its own `CLAUDE.md`, `README.md`, tests and `package.json`. Human docs: `README.md` (the site), `CONTRIBUTING.md` (how it works, adding a walkthrough, conventions).

## Commands

- `npm test` in the folder of the walkthrough you changed: must end with "all checks passed". A change isn't done until it passes. CI runs every walkthrough's tests (`.github/workflows/verify.yml`).
- `python3 -m http.server 8000` from the repository root to view pages (they load `../../assets/site.css`).

## Rules

- Colours only through the tokens in `assets/site.css` (light and dark). `tiny-cnn-walkthrough/index.html` has an inline copy: change both.
- Sentence case, plain copy, works at about 400px with no horizontal page scroll.
- External scripts only from jsDelivr or cdnjs, exact version, with an `integrity` hash.
- Never edit generated files by hand (each walkthrough's `CLAUDE.md` lists them); change the generator and rerun it.
- Every code cell a page shows has an `out` that the tests run and compare; a new cell needs one too.
- Third-party content (data, photos, weights) is credited on the page, in the walkthrough's `data/README.md` or `LICENSES.md`, and at the end of the root `LICENSE`.
- Adding a walkthrough or a section: follow `CONTRIBUTING.md` (section card, CI matrix, root README table).

## Repo etiquette

- IMPORTANT: work on `main`, but don't commit or push unless asked: the owner reviews every change in VS Code first.
- Commit subjects in the style of the history: "Adding X", "Fixing X".

## Environment

- Dev container on an ARM Mac; CI runs on x86. Floats on a rounding boundary can print differently in the last digit between the two.
- TensorFlow is in the user site-packages: run Python tools with `python3 -P`, not `-I`.
- `gh` is not logged in, and anonymous GitHub API calls hit the rate limit fast: ask the user for CI logs.
