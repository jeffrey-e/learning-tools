# CLAUDE.md

Guidance for Claude Code working in this repository. Each project folder has its own `CLAUDE.md` with the details.

## What this is

A static site of interactive learning pages, served by GitHub Pages from the `main` branch root at https://jeffrey-e.github.io/learning-tools/. There is no build step.

```
index.html                     home page: one card per section
assets/site.css                shared design system: tokens (light/dark), layout, components
ml-ai/index.html               section page: one card per walkthrough
ml-ai/tiny-cnn-walkthrough/    a self-contained project (page, tests, Python, its own package.json)
ml-ai/tiny-llm-walkthrough/    a self-contained project
ml-ai/log-autoencoder-walkthrough/  a self-contained project (Keras autoencoder on SSH logs)
ml-ai/image-recognition-walkthrough/  a self-contained project (husky or malamute: CNN vs transfer learning, TF.js)
.github/workflows/verify.yml   runs `npm test` in every project folder (matrix)
```

## Adding a page

1. Create a project folder under a section, with its own `package.json` (`npm test`), `requirements.txt`, `README.md` and `CLAUDE.md`. Use `ml-ai/tiny-llm-walkthrough/` as the template.
2. Link `assets/site.css` (relative path) and the same Google Fonts link; add a breadcrumb back to the section and home.
3. Add a card to the section's `index.html`. For a new section, add a folder with an `index.html` and a card on the home page.
4. Add the folder to the matrix in `.github/workflows/verify.yml`.

External tools go in the section page's "Other useful tools" list: one `<li>` card with an `<h3>` title, a sentence on what it shows, and a `.url` line naming the site and its makers.

## Conventions

- **Styles:** colours only through the tokens in `assets/site.css`, with light and dark values. `tiny-cnn-walkthrough/index.html` keeps its own inline copy (it's designed to work as a single downloaded file); if you change a token or shared component, change both.
- Sentence case for labels and headings. Plain, direct copy. Works at about 400px wide with no horizontal page scroll.
- Libraries only from jsDelivr or cdnjs, pinned to an exact version.
- Generated files (`weights.js`, `python/`, `notebooks/`) are never edited by hand.
- Run `npm test` in the project folder after every change; a change is not done until it passes.
