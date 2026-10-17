# Data

- `candidates.json`: every freely licensed candidate photo found on Wikimedia Commons, with its licence, author and page (`tools/fetch_commons.py`). The downloaded files (`candidates/`) are not committed.
- `labels.json`: the label of every candidate (husky, malamute or drop), from the automatic and visual first pass and the human review (`"by": "user"`).
- `photos.json`: the 404 photos used, with their split, photo session and credits (`tools/prepare_data.py`). The photos in `../images/` are resized and padded versions of the originals; each keeps its own licence (CC0, public domain, CC BY or CC BY-SA), see `licence` and `page`.
- `experiments.json`: every tuning experiment and the chosen runs (`tools/experiments.py`).
