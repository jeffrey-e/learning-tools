"""Turns the reviewed candidates into the dataset: resized photos in images/ and data/photos.json.

- Keeps the photos labelled husky or malamute in data/labels.json (first pass + human review).
- Groups photos by session: the same author on the same day (from the Commons metadata). A group
  never spans two splits, so the test set can't contain another shot of a training dog.
- Splits each breed by group: about 20% test, 15% validation, the rest training (seeded).
- Pads each photo to a square with grey (so no part of the dog is cut off) and resizes it to 224x224.

Usage:  python3 -P tools/prepare_data.py
"""
import json
import pathlib
import random
import re
import time
import urllib.parse
import urllib.request

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "https://commons.wikimedia.org/w/api.php"
UA = "learning-tools image-recognition-walkthrough (https://github.com/jeffrey-e/learning-tools)"
SIZE = 224
SEED = 1
TEST, VAL = 0.20, 0.15


def plain(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def commons_meta(titles):
    """Author and date of each file (for grouping photos from one session)."""
    meta = {}
    for i in range(0, len(titles), 50):
        q = {"action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "imageinfo", "iiprop": "extmetadata|user|timestamp",
             "format": "json", "formatversion": "2"}
        req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            for page in json.load(r)["query"]["pages"]:
                info = page["imageinfo"][0]
                em = info.get("extmetadata", {})
                date = plain(em.get("DateTimeOriginal", {}).get("value")) or plain(em.get("DateTime", {}).get("value")) or info["timestamp"]
                meta[page["title"]] = {"uploader": info.get("user", ""), "date": date}
        time.sleep(0.2)
    return meta


def letterbox(path):
    im = Image.open(path).convert("RGB")
    side = max(im.size)
    sq = Image.new("RGB", (side, side), (128, 128, 128))
    sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return sq.resize((SIZE, SIZE), Image.LANCZOS)


def main():
    cands = {c["id"]: c for c in json.loads((ROOT / "data" / "candidates.json").read_text())}
    labels = json.loads((ROOT / "data" / "labels.json").read_text())
    keep = sorted(k for k, v in labels.items() if v["label"] in ("husky", "malamute"))
    meta = commons_meta([cands[k]["title"] for k in keep])

    photos = []
    for k in keep:
        c, m = cands[k], meta[cands[k]["title"]]
        day = re.search(r"\d{4}[-:]\d{2}[-:]\d{2}", m["date"])
        photos.append({
            "id": k, "label": labels[k]["label"], "file": f"images/{k}.jpg",
            "group": f"{m['uploader']}|{day.group(0).replace(':', '-') if day else 'undated:' + c['title']}",
            "title": c["title"], "author": c["author"] or m["uploader"], "licence": c["licence"],
            "licence_url": c["licence_url"], "page": c["page"],
        })

    rng = random.Random(SEED)
    for breed in ("husky", "malamute"):
        groups = {}
        for p in photos:
            if p["label"] == breed:
                groups.setdefault(p["group"], []).append(p)
        order = sorted(groups)
        rng.shuffle(order)
        n, done = sum(len(g) for g in groups.values()), 0
        for g in order:
            split = "test" if done < TEST * n else "val" if done < (TEST + VAL) * n else "train"
            for p in groups[g]:
                p["split"] = split
            done += len(groups[g])

    (ROOT / "images").mkdir(exist_ok=True)
    for p in photos:
        letterbox(ROOT / "data" / "candidates" / (p["id"] + ".jpg")).save(ROOT / p["file"], quality=90)
    (ROOT / "data" / "photos.json").write_text(json.dumps(photos, indent=1, ensure_ascii=False) + "\n")

    for breed in ("husky", "malamute"):
        print(breed, {s: sum(p["label"] == breed and p["split"] == s for p in photos) for s in ("train", "val", "test")})
    sizes = {}
    for p in photos:
        sizes[p["group"]] = sizes.get(p["group"], 0) + 1
    print(len(sizes), "groups; largest:", sorted(sizes.values())[-5:])


if __name__ == "__main__":
    main()
