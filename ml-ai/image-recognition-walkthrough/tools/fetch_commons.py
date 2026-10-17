"""Collects candidate photos of Siberian Huskies and Alaskan Malamutes from Wikimedia Commons.

For every file: the breed it was found under (category or search), its licence, author and page,
and a 500px thumbnail in data/candidates/ (not committed). Only freely licensed files are kept:
CC0, public domain, CC BY and CC BY-SA. The labels are a first guess; tools/review.html is where
they get checked before training.

Usage:  python3 -P tools/fetch_commons.py
"""
import html
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "candidates"
API = "https://commons.wikimedia.org/w/api.php"
UA = "learning-tools image-recognition-walkthrough (https://github.com/jeffrey-e/learning-tools)"
THUMB = 500   # a standard Commons thumbnail width

SOURCES = {
    "husky": {
        "categories": ["Siberian Husky", "Siberian Husky heads", "Huskies in Germany", "Huskies in the Philippines",
                       "Huskies in Russia", "Huskies in the United States", "Siberian Husky puppies"],
        "search": [],
    },
    "malamute": {
        "categories": ["Alaskan Malamute", "Alaskan Malamute heads", "Alaskan Malamute puppies"],
        "search": ["Alaskan Malamute"],
    },
}


def api(**params):
    params |= {"format": "json", "formatversion": "2"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def category_files(cat):
    titles, cont = [], {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle="Category:" + cat, cmtype="file", cmlimit=500, **cont)
        titles += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return titles
        cont = d["continue"]


def search_files(q):
    titles, cont = [], {}
    while True:
        d = api(action="query", list="search", srsearch=q, srnamespace=6, srlimit=500, **cont)
        titles += [m["title"] for m in d["query"]["search"]]
        if "continue" not in d:
            return titles
        cont = d["continue"]


def plain(s):
    """Strips the HTML Commons puts in metadata fields."""
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def free(licence):
    lic = licence.lower()
    if "nc" in re.split(r"[\s-]+", lic) or "nd" in re.split(r"[\s-]+", lic):
        return False
    return lic.startswith(("cc0", "public domain", "pd", "cc by", "cc-by"))


def main():
    found = {}   # title -> {breeds, sources}
    for breed, src in SOURCES.items():
        for cat in src["categories"]:
            for t in category_files(cat):
                found.setdefault(t, {"breeds": set(), "sources": set()})
                found[t]["breeds"].add(breed)
                found[t]["sources"].add("category: " + cat)
        for q in src["search"]:
            for t in search_files(q):
                found.setdefault(t, {"breeds": set(), "sources": set()})
                found[t]["breeds"].add(breed)
                found[t]["sources"].add("search: " + q)
    print(f"{len(found)} files found")

    OUT.mkdir(parents=True, exist_ok=True)
    titles, rows = sorted(found), []
    for i in range(0, len(titles), 50):
        d = api(action="query", titles="|".join(titles[i:i + 50]), prop="imageinfo",
                iiprop="url|mime|size|extmetadata", iiurlwidth=THUMB)
        for page in d["query"]["pages"]:
            info = (page.get("imageinfo") or [None])[0]
            if not info or info["mime"] not in ("image/jpeg", "image/png", "image/webp"):
                continue
            meta = info.get("extmetadata", {})
            licence = plain(meta.get("LicenseShortName", {}).get("value"))
            if not free(licence) or min(info["width"], info["height"]) < 300:
                continue
            f = found[page["title"]]
            rows.append({
                "title": page["title"],
                "guess": sorted(f["breeds"])[0] if len(f["breeds"]) == 1 else "both",
                "sources": sorted(f["sources"]),
                "licence": licence,
                "licence_url": plain(meta.get("LicenseUrl", {}).get("value")),
                "author": plain(meta.get("Artist", {}).get("value"))[:200],
                "page": info["descriptionurl"],
                "thumb_url": info.get("thumburl", info["url"]),
                "width": info["width"], "height": info["height"],
            })
        time.sleep(0.2)

    for n, row in enumerate(rows):
        row["id"] = f"c{n:04d}"
        path = OUT / (row["id"] + ".jpg")
        if not path.exists():
            req = urllib.request.Request(row["thumb_url"], headers={"User-Agent": UA})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        path.write_bytes(r.read())
                    break
                except Exception as e:   # rate limited or a network hiccup: wait and retry
                    print("retry", row["id"], e)
                    time.sleep(5 * (attempt + 1))
            time.sleep(0.1)
    (ROOT / "data" / "candidates.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    by = {}
    for r in rows:
        by[r["guess"]] = by.get(r["guess"], 0) + 1
    print(f"{len(rows)} freely licensed candidates: {by}")


if __name__ == "__main__":
    main()
